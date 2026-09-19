import Diffusion
import Foundation
import NNC

public enum MiniMaxH3AudioConditioningError: Error, LocalizedError {
  case invalidDuration
  case invalidWeights(String)
  case nonFiniteOutput

  public var errorDescription: String? {
    switch self {
    case .invalidDuration:
      return "H3 reference audio must contain between 2 and 15 seconds of audio."
    case .invalidWeights(let detail):
      return
        "Cannot load H3 AudioVAE encoder: \(detail). Supply an H3 VAE checkpoint or the original audio_vae directory with --audio-encoder-file."
    case .nonFiniteOutput:
      return "H3 AudioVAE encoder produced non-finite reference latents."
    }
  }
}

/// H3 uses its AudioVAE posterior mean, not Whisper features or the output soundtrack.
public struct MiniMaxH3AudioConditioningEncoder {
  private let directory: URL?
  private let checkpoint: (path: String, mean: [Float], std: [Float])?

  public init(directory: URL) {
    self.directory = directory
    self.checkpoint = nil
  }

  public init(filePath: String, latentsMean: [Float], latentsStd: [Float]) {
    self.directory = nil
    self.checkpoint = (filePath, latentsMean, latentsStd)
  }

  public func encode(contentsOf path: String) throws -> Tensor<FloatType> {
    let channels = try AudioInput.readChannels(
      contentsOf: path, sampleRate: 32_000,
      channelCount: 2, maximumFrames: 480_000)
    return try encode(channels: channels)
  }

  public func encode(channels: [[Float]]) throws -> Tensor<FloatType> {
    let mean: [Double]
    let std: [Double]
    let archives: [SafeTensors]
    if let checkpoint {
      mean = checkpoint.mean.map(Double.init)
      std = checkpoint.std.map(Double.init)
      archives = []
    } else if let directory {
      let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
      let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
      mean = config?["latents_mean"] as? [Double] ?? []
      std = config?["latents_std"] as? [Double] ?? []
      let files = try FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: nil
      ).filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
      archives = try files.map { url in
        guard let archive = SafeTensors(url: url) else {
          throw MiniMaxH3AudioConditioningError.invalidWeights(url.lastPathComponent)
        }
        return archive
      }
    } else {
      throw MiniMaxH3AudioConditioningError.invalidWeights("missing weights")
    }
    guard mean.count == 32, std.count == 32, mean.allSatisfy(\.isFinite),
      std.allSatisfy({ $0.isFinite && $0 > 0 })
    else {
      throw MiniMaxH3AudioConditioningError.invalidWeights("invalid latent normalization")
    }
    func read(_ name: String) throws -> Tensor<Float> {
      for archive in archives {
        if let descriptor = archive.states[name] {
          return try archive.with(descriptor) { Tensor<Float>(from: $0) }
        }
      }
      throw MiniMaxH3AudioConditioningError.invalidWeights("missing \(name)")
    }
    let samples = (channels[0].count + 799) / 800 * 800
    var pcm = Tensor<Float>(Array(repeating: 0, count: 2 * samples), .CPU, .NCHW(2, 1, 1, samples))
    for channel in 0..<2 {
      for index in channels[channel].indices {
        pcm[channel, 0, 0, index] = channels[channel][index]
      }
    }
    let graph = DynamicGraph()
    return try graph.withNoGrad {
      let (mapper, model) = MiniMaxH3AudioEncoder(samples: samples)
      let input = graph.variable(pcm.toGPU(0))
      let length = samples / 800
      var mask = Tensor<Float>(
        Array(repeating: 0, count: length * length), .CPU, .NHWC(1, 1, length, length))
      for row in 0..<length {
        for column in (row + 1)..<length {
          mask[0, 0, row, column] = -Float.greatestFiniteMagnitude
        }
      }
      let causalMask = graph.variable(mask.toGPU(0))
      model.compile(inputs: input, causalMask)
      func parameterTensor(_ key: String) throws -> Tensor<Float> {
        var tensor: Tensor<Float>
        if key.hasPrefix("encoder."), key.hasSuffix(".weight") {
          let prefix = String(key.dropLast(7))
          let vector = try read("\(prefix).weight_v")
          let magnitude = try read("\(prefix).weight_g")
          let shape = vector.shape
          guard shape.count == 3, magnitude.shape.reduce(1, *) == shape[0] else {
            throw MiniMaxH3AudioConditioningError.invalidWeights(key)
          }
          let count = shape[1] * shape[2]
          var flat = vector.reshaped(.NC(shape[0], count)).copied()
          let scales = magnitude.reshaped(.C(shape[0]))
          try flat.withUnsafeMutableBytes { bytes in
            let values = bytes.bindMemory(to: Float.self)
            for output in 0..<shape[0] {
              let offset = output * count
              var norm: Double = 0
              for index in 0..<count {
                let value = Double(values[offset + index])
                norm += value * value
              }
              guard norm > 0 else { throw MiniMaxH3AudioConditioningError.invalidWeights(key) }
              let scale = scales[output] / Float(norm.squareRoot())
              for index in 0..<count { values[offset + index] *= scale }
            }
          }
          tensor = flat.reshaped(.NCHW(shape[0], shape[1], 1, shape[2]))
        } else if key == "pre_block.attn.qkv.bias" {
          var bias = Tensor<Float>(.CPU, .C(6_144))
          for (index, name) in ["q_bias", "zero_k_bias", "v_bias"].enumerated() {
            bias[(index * 2_048)..<((index + 1) * 2_048)] = try read("pre_block.attn.\(name)")
              .reshaped(.C(2_048))
          }
          tensor = bias
        } else {
          tensor = try read(key)
          if key.hasSuffix(".alpha") {
            tensor = tensor.reshaped(.NCHW(1, tensor.shape.reduce(1, *), 1, 1))
          } else if key == "mean_proj.weight" {
            tensor = tensor.reshaped(.NCHW(32, 32, 1, 1))
          } else if key.hasSuffix("_bias") {
            tensor = tensor.reshaped(.HWC(1, 1, 2_048))
          }
        }
        return tensor
      }
      if let checkpoint {
        try graph.openStore(
          checkpoint.path, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: checkpoint.path)
        ) { store in
          try store.read(
            "audio_encoder", model: model, strict: true,
            codec: [.jit, .q6p, .q8p, .i8x, .ezm7, .externalData])
        }
      } else {
        let parameterKeys = Dictionary(
          uniqueKeysWithValues: mapper(.generativeModels).flatMap { key, names in
            names.map { ("__audio_encoder__[\($0)]", key) }
          })
        var loadingError: Error?
        try graph.openStore(":memory:") { store in
          do {
            try store.read("audio_encoder", model: model, strict: true) { name, _, format, shape in
              guard loadingError == nil else { return .fail }
              do {
                guard let key = parameterKeys[name] else {
                  throw MiniMaxH3AudioConditioningError.invalidWeights("unmapped \(name)")
                }
                let tensor = try parameterTensor(key)
                guard tensor.shape.reduce(1, *) == shape.reduce(1, *) else {
                  throw MiniMaxH3AudioConditioningError.invalidWeights("shape mismatch for \(key)")
                }
                return .final(tensor.reshaped(format: format, shape: shape))
              } catch {
                loadingError = error
                return .fail
              }
            }
          } catch {
            throw loadingError ?? error
          }
        }
      }
      let encoded = model(inputs: input, causalMask)[0].as(of: Float.self).toCPU().rawValue
      var result = Tensor<FloatType>(.CPU, .HWC(1, 2 * length, 32))
      for channel in 0..<2 {
        for time in 0..<length {
          for feature in 0..<32 {
            let value =
              (encoded[channel, time, feature] - Float(mean[feature])) / Float(std[feature])
            guard value.isFinite, FloatType(value).isFinite else {
              throw MiniMaxH3AudioConditioningError.nonFiniteOutput
            }
            result[0, channel * length + time, feature] = FloatType(value)
          }
        }
      }
      return result
    }
  }
}
