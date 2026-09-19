# H3 audio references

`draw-things-cli generate --audio` accepts a single 2–15 second audio reference for
MiniMax H3 Ref2VA, optionally together with one or more `--image` references. This conditions new video and audio;
the input waveform is not muxed into the output as it is for LongCat.

The default uses the audio encoder in the model's Draw Things VAE checkpoint
(`minimax_h3_vae_f16.ckpt` for the standard H3 models). No separate encoder download
is needed. To override it, pass `--audio-encoder-file` with a native `.ckpt` path
or the original MiniMax-H3 `audio_vae` directory containing `config.json` and
`diffusion_pytorch_model.safetensors`. Relative paths resolve within `--models-dir`.
The original weights are at https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/audio_vae.

```sh
draw-things-cli generate \
  --models-dir "/path/to/Draw Things Models" \
  --model minimax_h3_ref2va_q8p.ckpt \
  --image reference.png --audio reference.wav \
  --prompt "The character in Picture 1 speaks in the voice of Audio 1." \
  --config-json '{"loras":[{"file":"YOUR_LORA.ckpt","weight":1,"version":"minimax_h3"}]}' \
  --steps 8 --seed 42 --frames 56 --width 512 --height 512 \
  --output reference-result.mp4
```

The port follows antirez/h3.c commit
`8974cc055ea9c02fcd14cc27dfda3e1027c05153`: 32 kHz stereo, zero-padding to an
800-sample boundary, AudioVAE posterior mean and latent normalization, 0.999 clean
reference plus 0.001 noise, reference-audio timestep 1, and stereo rotary positions.
It preserves Draw Things' existing image-reference ordering and uses the audio
reference after the image references. Multiple audio references and video-reference
input are not exposed by this CLI path.

After rebasing onto upstream native audio support, the CLI passes stereo PCM through
the shared audio-conditioning context. H3 uses the h3.c-validated posterior-mean
encoder here; the upstream sampler and denoiser APIs carry the resulting references.
LongCat keeps its upstream waveform, loudness-normalization and cancellation paths.

## Verification

`bazel test //Libraries/AudioConverter:MiniMaxH3AudioReferenceTests` runs the focused tests.
For encoder parity, set `H3_AUDIO_VAE_DIRECTORY` to the upstream weights directory
and `H3_AUDIO_ORACLE` to an h3.c Float32 output fixture with layout [32, 2, 80].
The fixture input is two seconds at 32 kHz: left is a 220 Hz sine and right is a
330 Hz sine, both amplitude 0.1. Set `H3_AUDIO_VAE_CHECKPOINT` to additionally
compare a native Draw Things VAE checkpoint with the original weights.

On Xcode 27, the pinned GPU library's MFA shaders can fail Metal compilation.
`--test_env=H3_TEST_DISABLE_MFA=1` runs these tests on the MPS fallback backend.
This test option does not change the CLI's production GPU backend. Encoder parity
on that fallback measured relative L2 error 0.000208 against h3.c; the native FP16
checkpoint differed from the original weights by 0.000543.

## h3.c license

MIT License

Copyright (c) 2026 Salvatore Sanfilippo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
