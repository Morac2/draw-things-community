// MiniMax H3 video autoencoder graph definitions.
import Foundation
import NNC

/// AudioVAE posterior mean before latent normalization. Input is stereo NCHW
/// [2, 1, 1, samples], zero-padded to a multiple of 800; output is [2, T, 32].
/// The encoder uses Snake1d (not the decoder's anti-aliased SnakeBeta).
/// Ported against antirez/h3.c; see Libraries/AudioConverter/H3_AUDIO_REFERENCE.md.
public func MiniMaxH3AudioEncoder(samples: Int) -> (ModelWeightMapper, Model) {
  precondition(samples > 0 && samples % 800 == 0)
  let x = Input()
  let causalMask = Input()
  var parameters = [(String, Model.Parameters)]()
  func convolution(
    _ prefix: String, channels: Int, kernel: Int, stride: Int = 1,
    dilation: Int = 1, padding: Int, name: String
  ) -> Model {
    let conv = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernel], dilation: [1, dilation],
      hint: Hint(
        stride: [1, stride],
        border: Hint.Border(begin: [0, padding], end: [0, padding])), name: name)
    parameters.append(("\(prefix).weight", conv.weight))
    parameters.append(("\(prefix).bias", conv.bias))
    return conv
  }
  func snake(_ input: Model.IO, channels: Int, prefix: String, name: String) -> Model.IO {
    let alpha = Parameter<Float>(
      .GPU(0), .NCHW(1, channels, 1, 1),
      trainable: false, name: "\(name)_alpha")
    parameters.append(("\(prefix).alpha", alpha.weight))
    return input + (input .* alpha).sin().pow(2) .* (1 / (alpha + 1e-9))
  }
  func dense(_ prefix: String, count: Int, noBias: Bool = false, name: String) -> Model {
    let layer = Dense(count: count, noBias: noBias, name: name)
    parameters.append(("\(prefix).weight", layer.weight))
    if !noBias { parameters.append(("\(prefix).bias", layer.bias)) }
    return layer
  }
  func norm(_ prefix: String, name: String) -> Model {
    let layer = LayerNorm(epsilon: 1e-5, axis: [2], name: name)
    parameters.append(("\(prefix).weight", layer.weight))
    parameters.append(("\(prefix).bias", layer.bias))
    return layer
  }
  var out = convolution(
    "encoder.block.0", channels: 64, kernel: 7, padding: 3, name: "encoder_input")(x)
  var channels = 64
  for (index, stride) in [2, 4, 4, 5, 5].enumerated() {
    let prefix = "encoder.block.\(index + 1).block"
    for (residual, dilation) in [1, 3, 9].enumerated() {
      let block = "\(prefix).\(residual).block"
      var branch = snake(
        out, channels: channels, prefix: "\(block).0",
        name: "encoder_block_\(index)_res\(residual)_snake1")
      branch = convolution(
        "\(block).1", channels: channels, kernel: 7,
        dilation: dilation, padding: 3 * dilation,
        name: "encoder_block_\(index)_res\(residual)_conv1")(branch)
      branch = snake(
        branch, channels: channels, prefix: "\(block).2",
        name: "encoder_block_\(index)_res\(residual)_snake2")
      out =
        out
        + convolution(
          "\(block).3", channels: channels, kernel: 1, padding: 0,
          name: "encoder_block_\(index)_res\(residual)_conv2")(branch)
    }
    out = snake(
      out, channels: channels, prefix: "\(prefix).3", name: "encoder_block_\(index)_snake")
    channels *= 2
    out = convolution(
      "\(prefix).4", channels: channels, kernel: 2 * stride,
      stride: stride, padding: (stride + 1) / 2, name: "encoder_block_\(index)_downsample")(out)
  }
  out = snake(out, channels: channels, prefix: "encoder.block.6", name: "encoder_final_snake")
  out = convolution(
    "encoder.block.7", channels: channels, kernel: 3, padding: 1, name: "encoder_final_conv")(out)
  let length = samples / 800
  let hidden = out.reshaped([2, 2_048, length]).permuted(0, 2, 1).contiguous()
  var base = dense("pre_block.proj", count: 32, name: "input_projection")(
    norm("pre_block.norm3", name: "norm3")(hidden))
  let normalized = norm("pre_block.norm1", name: "norm1")(hidden)
  let qkv = dense("pre_block.attn.qkv", count: 6_144, name: "qkv")(normalized)
  let projections = (0..<3).map { index in
    qkv.reshaped(
      [2, length, 2_048], offset: [0, 0, index * 2_048],
      strides: [length * 6_144, 6_144, 1]
    ).contiguous()
      .reshaped(.NHWC(2, length, 8, 256))
  }
  // Explicit masking also preserves causality on the MPS fallback backend.
  let scores =
    Matmul(transposeB: (2, 3))(
      (1.0 / 16) * projections[0].transposed(1, 2), projections[1].transposed(1, 2)) + causalMask
  let probabilities = scores.reshaped([2 * 8 * length, length]).softmax()
    .reshaped([2, 8, length, length])
  let attended = Matmul()(probabilities, projections[2].transposed(1, 2))
    .transposed(1, 2).contiguous()
  // Pool contiguous groups of eight within each head, then average the heads.
  let pooled = attended.reshaped([2, length, 8, 32, 8])
    .reduced(.mean, axis: [2, 4]).reshaped([2, length, 32])
  base = base + dense("pre_block.attn.proj", count: 32, name: "attention_output")(pooled)
  let mlp = norm("pre_block.mlp.norm", name: "mlp_norm")(
    norm("pre_block.norm2", name: "norm2")(base))
  let gate = dense("pre_block.mlp.w0", count: 64, name: "mlp_gate")(mlp).GELU(approximate: .tanh)
  let value = dense("pre_block.mlp.w1", count: 64, name: "mlp_up")(mlp)
  base = base + dense("pre_block.mlp.w2", count: 32, name: "mlp_down")(gate .* value)
  let result = dense("mean_proj", count: 32, name: "mean_proj")(base)
  return (
    { _ in
      Dictionary(uniqueKeysWithValues: parameters.map { ($0.0, ModelWeightElement([$0.1.name])) })
    }, Model([x, causalMask], [result])
  )
}

private func MiniMaxH3VideoEncoderResnetBlock(
  inChannels: Int, outChannels: Int, frames: Int, height: Int, width: Int
) -> Model {
  let x = Input()
  let norm1 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2], name: "norm1")
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1]), format: .OIHW, name: "conv1")
  var out = conv1(
    norm1(x).swish().padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0])
      .padded(.zero, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
      .reshaped([1, frames + 2, height + 2, width + 2, inChannels])
  ).reshaped(.NHWC(frames, height, width, outChannels))
  let norm2 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2], name: "norm2")
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1]), format: .OIHW, name: "conv2")
  out = conv2(
    norm2(out).swish().padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0])
      .padded(.zero, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
      .reshaped([1, frames + 2, height + 2, width + 2, outChannels])
  ).reshaped(.NHWC(frames, height, width, outChannels))
  if inChannels != outChannels {
    let shortcut = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1],
      hint: Hint(stride: [1, 1, 1]), format: .OIHW, name: "nin_shortcut")
    out =
      shortcut(x.reshaped([1, frames, height, width, inChannels]))
      .reshaped(.NHWC(frames, height, width, outChannels)) + out
  } else {
    out = x + out
  }
  return Model([x], [out])
}

public func MiniMaxH3VideoEncoder(frames: Int, height: Int, width: Int) -> Model {
  precondition(frames > 0 && height % 16 == 0 && width % 16 == 0)
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1]), format: .OIHW, name: "conv_in")
  var out = convIn(
    x.padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0])
      .padded(.zero, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
      .reshaped([1, frames + 2, height + 2, width + 2, 3])
  ).reshaped(.NHWC(frames, height, width, 128))
  var channels = 128
  var frames = frames
  var height = height
  var width = width
  for (level, outChannels) in [128, 256, 256, 512, 512, 1_024].enumerated() {
    for _ in 0..<2 {
      out = MiniMaxH3VideoEncoderResnetBlock(
        inChannels: channels, outChannels: outChannels, frames: frames, height: height, width: width
      )(out)
      channels = outChannels
    }
    if level < 4 {
      let temporalStride = level == 1 || level == 2 ? 2 : 1
      let downsample = Convolution(
        groups: 1, filters: channels, filterSize: [3, 3, 3],
        hint: Hint(stride: [temporalStride, 2, 2]), format: .OIHW, name: "downsample_\(level)")
      out = downsample(
        out.padded(.reflect, begin: [0, 0, 0, 0], end: [0, 1, 1, 0])
          .padded(.zero, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
          .reshaped([1, frames + 2, height + 1, width + 1, channels]))
      frames = (frames + temporalStride - 1) / temporalStride
      height /= 2
      width /= 2
      out = out.reshaped(.NHWC(frames, height, width, channels))
    }
  }
  let normOut = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2], name: "norm_out")
  let convOut = Convolution(
    groups: 1, filters: 48, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1]), format: .OIHW, name: "conv_out")
  out = convOut(
    normOut(out).swish().padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0])
      .padded(.zero, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
      .reshaped([1, frames + 2, height + 2, width + 2, channels]))
  let quant = Convolution(
    groups: 1, filters: 48, filterSize: [1, 1, 1],
    hint: Hint(stride: [1, 1, 1]), format: .OIHW, name: "quant_conv")
  out = quant(out).reshaped(.NHWC(frames, height, width, 48))
  return Model([x], [out])
}

func MiniMaxH3VideoDecoderRotaryEmbedding(latentFrames: Int, latentHeight: Int, latentWidth: Int)
  -> Tensor<Float>
{
  let patchCount = latentFrames * latentHeight * latentWidth
  let suffixCount = 4 + 1
  var rotary = Tensor<Float>(
    .CPU, .NHWC(1, patchCount + suffixCount, 1, 64))
  let inverseFrequencies = (0..<8).map { index in
    pow(100.0, -Double(index) / 8)
  }
  var row = 0
  for frame in 0..<latentFrames {
    let t = 2 * (Double(frame) + 0.5) / Double(latentFrames) - 1
    for yIndex in 0..<latentHeight {
      let y = 2 * (Double(yIndex) + 0.5) / Double(latentHeight) - 1
      for xIndex in 0..<latentWidth {
        let x = 2 * (Double(xIndex) + 0.5) / Double(latentWidth) - 1
        var angleIndex = 0
        for coordinate in [t, y, x] {
          for inverseFrequency in inverseFrequencies {
            let angle = 2 * Double.pi * coordinate * inverseFrequency
            rotary[0, row, 0, 2 * angleIndex] = Float(cos(angle))
            rotary[0, row, 0, 2 * angleIndex + 1] = Float(sin(angle))
            angleIndex += 1
          }
        }
        for index in 48..<64 {
          rotary[0, row, 0, index] = index % 2 == 0 ? 1 : 0
        }
        row += 1
      }
    }
  }
  for suffix in 0..<suffixCount {
    for index in 0..<64 {
      rotary[0, patchCount + suffix, 0, index] = index % 2 == 0 ? 1 : 0
    }
  }
  return rotary
}

func H3VideoDecoderAttention(sequenceLength: Int, hiddenSize: Int) -> Model {
  let x = Input()
  let rot = Input()
  let q = Dense(count: hiddenSize, name: "to_q")
  let k = Dense(count: hiddenSize, name: "to_k")
  let v = Dense(count: hiddenSize, name: "to_v")
  let qNorm = RMSNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  let kNorm = RMSNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  var queries = q(x).reshaped(.NHWC(1, sequenceLength, hiddenSize / 64, 64))
  var keys = k(x).reshaped(.NHWC(1, sequenceLength, hiddenSize / 64, 64))
  let values = v(x).reshaped(.NHWC(1, sequenceLength, hiddenSize / 64, 64))
  queries = Functional.cmul(left: qNorm(queries), right: rot)
  keys = Functional.cmul(left: kNorm(keys), right: rot)
  let attention = ScaledDotProductAttention(
    scale: 1 / Float(64).squareRoot(), flags: [.Float16])
  let attended = attention(queries, keys, values).reshaped([
    1, sequenceLength, hiddenSize,
  ])
  let output = Dense(count: hiddenSize, name: "to_out")
  return Model([x, rot], [output(attended)])
}

func H3VideoDecoderBlock(sequenceLength: Int, hiddenSize: Int) -> Model {
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm1")
  let attention = H3VideoDecoderAttention(sequenceLength: sequenceLength, hiddenSize: hiddenSize)
  let scale1 = Parameter<Float>(
    .GPU(0), .HWC(1, 1, hiddenSize), name: "scale1")
  var out =
    x + attention(norm1(x).to(.Float16), rot).to(of: x) .* scale1
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm2")
  let up = Dense(count: 8_192, name: "ff_up")
  let gate = Dense(count: 8_192, name: "ff_gate")
  let down = Dense(count: hiddenSize, name: "ff_down")
  let normed = norm2(out).to(.Float16)
  let scale2 = Parameter<Float>(
    .GPU(0), .HWC(1, 1, hiddenSize), name: "scale2")
  out = residual + down(up(normed) .* gate(normed).swish()).to(of: residual) .* scale2
  return Model([x, rot], [out])
}

public func MiniMaxH3VideoDecoder(
  latentFrames: Int, latentHeight: Int, latentWidth: Int, hiddenSize: Int, layers: Int,
  includeHidden: Bool = false
) -> Model {
  let numPatches = latentFrames * latentHeight * latentWidth
  let latentTokens = Input()
  let rot = Input()
  let zeroToken = Input()
  let postQuant = Dense(count: 24, name: "post_quant_conv")
  let input = Dense(count: hiddenSize, name: "decoder_proj_in")
  var out = input(postQuant(latentTokens.reshaped(.HWC(1, numPatches, 24)))).to(.Float32)
  let registers = Parameter<Float>(
    .GPU(0), .HWC(1, 4, hiddenSize),
    name: "register_tokens")
  out = Functional.concat(axis: 1, out, registers, zeroToken)
  let sequenceLength = numPatches + 4 + 1
  for _ in 0..<layers {
    let block = H3VideoDecoderBlock(
      sequenceLength: sequenceLength, hiddenSize: hiddenSize)
    out = block(out, rot)
  }
  let hidden = out
  let norm = LayerNorm(epsilon: 1e-5, axis: [2], name: "decoder_norm_out")
  out = norm(out).to(of: latentTokens)
  let output = Dense(count: 3 * 4 * 16 * 16, name: "decoder_proj_out")
  out = output(out).reshaped(
    [
      1, numPatches,
      3 * 4 * 16 * 16,
    ],
    offset: [0, 0, 0],
    strides: [
      sequenceLength * 3 * 4 * 16 * 16,
      3 * 4 * 16 * 16, 1,
    ]
  ).contiguous()
  out = out.reshaped([latentFrames, latentHeight, latentWidth, 3, 4, 16, 16])
    .permuted(0, 4, 1, 5, 2, 6, 3).contiguous().reshaped(
      .NHWC((latentFrames - 1) * 4 + 1, latentHeight * 16, latentWidth * 16, 3),
      offset: [3, 0, 0, 0],
      strides: [latentHeight * 16 * latentWidth * 16 * 3, latentWidth * 16 * 3, 3, 1]
    ).contiguous()
  return Model([latentTokens, rot, zeroToken], includeHidden ? [out, hidden] : [out])
}
