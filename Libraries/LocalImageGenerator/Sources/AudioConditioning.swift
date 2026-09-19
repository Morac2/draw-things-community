import Atomics
import AudioConverter
import Diffusion
import Foundation
import ModelZoo
import NNC

// AVC retains full-span encoded conditioning across segment calls. Encoding happens inside
// the generation branch; its result is reused directly across hires passes.
final class AudioConditioningContext {
  let waveform: Tensor<Float>
  let encoderFilePath: String?
  let videoFrames: Int
  let zeroAudioFeatures: Bool
  var startFrame: Int = 0
  var encodedAudioCond: [Tensor<FloatType>]?

  init(
    waveform: Tensor<Float>, encoderFilePath: String?, videoFrames: Int,
    zeroAudioFeatures: Bool = false
  ) {
    self.waveform = waveform
    self.encoderFilePath = encoderFilePath
    self.videoFrames = videoFrames
    self.zeroAudioFeatures = zeroAudioFeatures
  }
}

extension LocalImageGenerator {
  func encodeAudioCond(
    _ audio: AudioConditioningContext?, graph: DynamicGraph, model: String,
    videoFrames: Int, framesPerSecond: Int, cancellation: (@escaping () -> Void) -> Void
  ) -> [DynamicGraph.Tensor<FloatType>]? {
    guard let audio else { return [] }
    dispatchPrecondition(condition: .onQueue(queue))
    switch ModelZoo.versionForModel(model) {
    case .longcatVideoAvatar1_5:
      guard videoFrames > 0, videoFrames % 4 == 1 else { return nil }
      if audio.encodedAudioCond == nil {
        let features: LongCatAudioFeatures
        if audio.zeroAudioFeatures {
          features = .zero(
            videoFrames: audio.videoFrames, framesPerSecond: framesPerSecond)
        } else {
          guard let filePath = audio.encoderFilePath else { return nil }
          let isCancelled = ManagedAtomic(false)
          cancellation { isCancelled.store(true, ordering: .releasing) }
          guard
            let encoded = try? LongCatAudioConditioningEncoder(filePath: filePath).encode(
              audio.waveform, videoFrames: audio.videoFrames, framesPerSecond: framesPerSecond,
              shouldContinue: { !isCancelled.load(ordering: .acquiring) })
          else { return nil }
          features = encoded
        }
        audio.encodedAudioCond = features.conditioning().tensors
      }
      guard let encodedAudioCond = audio.encodedAudioCond else { return nil }
      return LongCatAudioConditioning(
        audioFirst: encodedAudioCond[0], audioLatter: encodedAudioCond[1]
      ).segment(startFrame: audio.startFrame, videoFrames: videoFrames)
        .tensors.map { graph.variable($0.toGPU(0)) }
    case .minimaxH3:
      guard audio.waveform.shape[0] == 1 || audio.waveform.shape[0] == 2 else { return nil }
      guard let filePath = audio.encoderFilePath else { return nil }
      let scaling = ModelZoo.latentsScalingForModel(model)
      guard let mean = scaling.audioMean, let std = scaling.audioStd,
        mean.count == 32, std.count == 32, std.allSatisfy({ $0 > 0 })
      else { return nil }
      let isCancelled = ManagedAtomic(false)
      cancellation { isCancelled.store(true, ordering: .releasing) }
      guard !isCancelled.load(ordering: .acquiring) else { return nil }
      let channels = audio.waveform.copied().withUnsafeBytes { bytes -> [[Float]] in
        let samples = bytes.bindMemory(to: Float.self)
        let length = audio.waveform.shape[1]
        return (0..<2).map { channel in
          let offset = min(channel, audio.waveform.shape[0] - 1) * length
          return Array(samples[offset..<(offset + length)])
        }
      }
      let encoder: MiniMaxH3AudioConditioningEncoder
      if URL(fileURLWithPath: filePath).pathExtension == "ckpt" {
        encoder = MiniMaxH3AudioConditioningEncoder(
          filePath: filePath, latentsMean: mean, latentsStd: std)
      } else {
        encoder = MiniMaxH3AudioConditioningEncoder(directory: URL(fileURLWithPath: filePath))
      }
      guard let latents = try? encoder.encode(channels: channels),
        !isCancelled.load(ordering: .acquiring)
      else { return nil }
      return [graph.variable(latents.toGPU(0))]
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v,
      .wurstchenStageC, .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large,
      .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage,
      .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b,
      .ideogram4, .krea2, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b:
      return nil
    }
  }
}
