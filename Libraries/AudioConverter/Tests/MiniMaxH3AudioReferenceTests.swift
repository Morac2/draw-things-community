import AudioConverter
import Diffusion
import Foundation
import NNC
import XCTest

final class MiniMaxH3AudioReferenceTests: XCTestCase {
  private var savedFlags = DynamicGraph.flags

  override func setUp() {
    super.setUp()
    savedFlags = DynamicGraph.flags
    // Xcode 27's Metal compiler currently rejects the pinned MFA shader library.
    // Opt into MPS only for testing that toolchain; keep the normal backend as default.
    if ProcessInfo.processInfo.environment["H3_TEST_DISABLE_MFA"] == "1" {
      DynamicGraph.flags.formUnion([.disableMFA, .disableMixedMPSGEMM, .disableMixedMPSSoftMax])
    }
  }

  override func tearDown() {
    DynamicGraph.flags = savedFlags
    super.tearDown()
  }

  func testQwen3AcceptsChannelFirstTokenEmbeddings() throws {
    guard ProcessInfo.processInfo.environment["H3_TEST_DISABLE_MFA"] != "1" else {
      throw XCTSkip("This regression exercises masked Metal attention and requires MFA.")
    }
    let graph = DynamicGraph()
    graph.withNoGrad {
      let tokens = graph.variable(Tensor<Int32>([0, 1, 2, 3], .CPU, .C(4)).toGPU(0))
      let rotary = graph.variable(
        QwenVLRotaryEmbedding(sequenceLength: 4, of: Float16.self).toGPU(0))
      var mask = Tensor<Float16>(Array(repeating: 0, count: 16), .CPU, .NHWC(1, 1, 4, 4))
      for row in 0..<4 {
        for column in (row + 1)..<4 { mask[0, 0, row, column] = -Float16.greatestFiniteMagnitude }
      }
      let causalMask = graph.variable(mask.toGPU(0))
      let embeddingMask = graph.variable(.GPU(0), .WC(4, 1), of: Float16.self)
      embeddingMask.full(1)
      let imageEmbedding = graph.variable(.GPU(0), .WC(4, 128), of: Float16.self)
      imageEmbedding.full(0)
      let model = Qwen3(
        Float16.self, vocabularySize: 8, width: 128, tokenLength: 4,
        layers: 1, MLP: 256, heads: 8, outputHiddenStates: [0], noFinalNormalizedOutput: true,
        batchSize: 1, usesFlashAttention: true, injectEmbeddings: true, deepStackLayers: 1)
      let output = model(
        inputs: tokens, rotary, causalMask, embeddingMask, imageEmbedding, imageEmbedding)[0]
        .as(of: Float16.self).toCPU().rawValue
      XCTAssertEqual(output.shape, [4, 128])
      output.withUnsafeBytes { bytes in
        XCTAssertTrue(bytes.bindMemory(to: Float16.self).allSatisfy { $0.isFinite })
      }
    }
  }

  func testReferenceAudioRotaryTimeline() {
    let rows = MiniMaxH3RotaryEmbedding(
      textLength: 3, audioLength: 4,
      videoFrames: 1, videoHeight: 4, videoWidth: 4,
      referenceImages: [(4, 4, 3)], referenceAudios: [(4, 4)], videoPosition: 6)
    XCTAssertEqual(rows.shape[1], 19)
    // Text (3), image (4), reference audio (4), target audio (4), video (4).
    for (row, position) in [(7, Float(4)), (8, 5), (9, 4), (10, 5), (11, 6), (15, 6)] {
      XCTAssertEqual(rows[0, row, 0, 0], cos(position), accuracy: 1e-6)
      XCTAssertEqual(rows[0, row, 0, 1], sin(position), accuracy: 1e-6)
    }
    // Stereo channels share time but retain different spatial positions.
    XCTAssertNotEqual(rows[0, 7, 0, 64], rows[0, 9, 0, 64])
  }

  func testRejectInvalidAudioBeforeLoadingWeights() {
    let encoder = MiniMaxH3AudioConditioningEncoder(directory: URL(fileURLWithPath: "/nonexistent"))
    for channels in [
      [[Float]](), [[Float](repeating: 0, count: 64_000)],
      [[Float](repeating: 0, count: 100), [Float](repeating: 0, count: 100)],
    ] {
      XCTAssertThrowsError(try encoder.encode(channels: channels)) { error in
        guard case MiniMaxH3AudioConditioningError.invalidDuration = error else {
          return XCTFail("Unexpected error: \(error)")
        }
      }
    }
  }

  func testReferenceAudioFixedGraphAndLoRAMapping() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let inputs: [DynamicGraph.AnyTensor] = [
        graph.variable(.GPU(0), .HWC(1, 3, 5_120), of: Float16.self),
        graph.variable(.GPU(0), .HWC(1, 4, 256), of: Float16.self),
        graph.variable(.GPU(0), .NHWC(1, 2, 2, 24), of: Float16.self),
        graph.variable(.GPU(0), .HWC(1, 4, 32), of: Float16.self),
      ]
      for input in inputs { input.as(of: Float16.self).full(0.01) }
      let (mapper, base) = MiniMaxH3Fixed(
        timesteps: 1, hiddenSize: 8, layers: 1,
        textLength: (0, 3), usesFlashAttention: .scale1, referenceImageCount: 1,
        referenceAudioCount: 1)
      let output = base(inputs: inputs[0], Array(inputs.dropFirst()))
      XCTAssertEqual(output.count, 37)  // context, image, audio, 30 modulations, 4 output modulations
      XCTAssertEqual(output[2].shape[1], 4)
      let (loraMapper, lora) = LoRAMiniMaxH3Fixed(
        timesteps: 1, hiddenSize: 8, layers: 1,
        textLength: (0, 3), usesFlashAttention: .scale1, referenceImageCount: 1,
        referenceAudioCount: 1,
        LoRAConfiguration: LoRANetworkConfiguration(rank: 0, scale: 1, highPrecision: false))
      lora.compile(inputs: inputs)
      let expected = mapper(.generativeModels)
      let actual = loraMapper(.generativeModels)
      XCTAssertEqual(Set(expected.keys), Set(actual.keys))
      for (key, names) in expected { XCTAssertEqual(Array(names), Array(actual[key]!), key) }
    }
  }

  func testEncoderAgainstH3C() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let directory = environment["H3_AUDIO_VAE_DIRECTORY"],
      let oraclePath = environment["H3_AUDIO_ORACLE"]
    else {
      throw XCTSkip("Set H3_AUDIO_VAE_DIRECTORY and H3_AUDIO_ORACLE for real-weight parity.")
    }
    let channels = (0..<2).map { channel in
      (0..<64_000).map { index in
        Float(0.1 * sin(2 * Double.pi * Double(channel == 0 ? 220 : 330) * Double(index) / 32_000))
      }
    }
    let actual = try MiniMaxH3AudioConditioningEncoder(directory: URL(fileURLWithPath: directory))
      .encode(channels: channels)
    if let checkpoint = environment["H3_AUDIO_VAE_CHECKPOINT"] {
      let config =
        try JSONSerialization.jsonObject(
          with: Data(
            contentsOf:
              URL(fileURLWithPath: directory).appendingPathComponent("config.json")))
        as! [String: Any]
      let native = try MiniMaxH3AudioConditioningEncoder(
        filePath: checkpoint,
        latentsMean: (config["latents_mean"] as! [Double]).map(Float.init),
        latentsStd: (config["latents_std"] as! [Double]).map(Float.init)
      ).encode(channels: channels)
      var error = 0.0
      var magnitude = 0.0
      for time in 0..<160 {
        for feature in 0..<32 {
          let a = Double(actual[0, time, feature])
          let b = Double(native[0, time, feature])
          error += (a - b) * (a - b)
          magnitude += a * a
        }
      }
      print("Native H3 checkpoint relative L2 error: \(sqrt(error / magnitude))")
      XCTAssertLessThan(sqrt(error / magnitude), 0.01)
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: oraclePath))
    XCTAssertEqual(data.count, 32 * 2 * 80 * 4)
    let expected = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    var squaredError: Double = 0
    var squaredReference: Double = 0
    for feature in 0..<32 {
      for channel in 0..<2 {
        for time in 0..<80 {
          let reference = Double(expected[(feature * 2 + channel) * 80 + time])
          let value = Double(actual[0, channel * 80 + time, feature])
          squaredError += (value - reference) * (value - reference)
          squaredReference += reference * reference
        }
      }
    }
    let relativeError = sqrt(squaredError / squaredReference)
    print("H3 AudioVAE relative L2 error: \(relativeError)")
    XCTAssertLessThan(relativeError, 0.002)
  }

  func testReferenceAudioReachesDenoiser() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let inputs: [DynamicGraph.AnyTensor] =
        [
          graph.variable(.GPU(0), .NHWC(1, 2, 2, 24), of: Float16.self),
          graph.variable(.GPU(0), .HWC(1, 4, 32), of: Float16.self),
          graph.variable(.GPU(0), .HWC(1, 3, 8), of: Float.self),
          graph.variable(
            Tensor<Float16>(
              from: MiniMaxH3RotaryEmbedding(
                textLength: 3, audioLength: 4, videoFrames: 1, videoHeight: 2, videoWidth: 2,
                referenceImages: [(2, 2, 3)], referenceAudios: [(4, 4)], videoPosition: 6)
            ).toGPU(0)),
          graph.variable(.GPU(0), .NHWC(1, 1, 1, 8), of: Float16.self),
          graph.variable(.GPU(0), .HWC(1, 4, 8), of: Float16.self),
        ] + (0..<34).map { _ in graph.variable(.GPU(0), .HWC(1, 1, 8), of: Float16.self) }
      for (index, input) in inputs.enumerated() where index != 3 {
        if index == 2 {
          input.as(of: Float.self).full(0.1)
        } else {
          input.as(of: Float16.self).full(0.1)
        }
      }
      let (mapper, model) = MiniMaxH3(
        hiddenSize: 8, layers: 1, textLength: 3,
        audioLength: 4, videoFrames: 1, videoHeight: 2, videoWidth: 2,
        usesFlashAttention: .scale1, referenceImageSizes: [(2, 2)], referenceAudioLengths: [4])
      let first = model(inputs: inputs[0], Array(inputs.dropFirst()))[0]
        .as(of: Float16.self).toCPU().rawValue.copied()
      inputs[5].as(of: Float16.self).randn()
      let second = model(inputs: inputs[0], Array(inputs.dropFirst()))[0]
        .as(of: Float16.self).toCPU().rawValue
      var difference: Float = 0
      for y in 0..<2 {
        for x in 0..<2 {
          for c in 0..<24 {
            let value = Float(second[0, y, x, c])
            XCTAssertTrue(value.isFinite)
            difference += abs(value - Float(first[0, y, x, c]))
          }
        }
      }
      XCTAssertGreaterThan(difference, 0)
      let (loraMapper, lora) = LoRAMiniMaxH3(
        hiddenSize: 8, layers: 1, textLength: 3,
        audioLength: 4, videoFrames: 1, videoHeight: 2, videoWidth: 2,
        usesFlashAttention: .scale1, referenceImageSizes: [(2, 2)], referenceAudioLengths: [4],
        LoRAConfiguration: LoRANetworkConfiguration(rank: 0, scale: 1, highPrecision: false))
      lora.compile(inputs: inputs)
      let expected = mapper(.generativeModels)
      let actual = loraMapper(.generativeModels)
      XCTAssertEqual(Set(expected.keys), Set(actual.keys))
      for (key, names) in expected { XCTAssertEqual(Array(names), Array(actual[key]!), key) }
    }
  }
}
