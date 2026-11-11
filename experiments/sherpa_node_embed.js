// Embed raw float32 waveforms (16 kHz, one file per utterance) with sherpa-onnx-node.
// Used by 30_sherpa_1x_check.py for sherpa-onnx 1.13.8, the newest release allowed by NexaRoom's
// requirement sherpa-onnx>=1.13,<2. Install in a separate folder with
//   npm install --ignore-scripts sherpa-onnx-node@1.13.8
// usage: node sherpa_node_embed.js <sherpa-onnx-node module dir> <model.onnx> <list.txt> <out.f32>
// list.txt: one input file per line; out.f32: embeddings concatenated in list order (float32).
const [modDir, model, listPath, out] = process.argv.slice(2);
const sherpa = require(modDir);
const fs = require('fs');
const ex = new sherpa.SpeakerEmbeddingExtractor({model: model, numThreads: 1, provider: 'cpu', debug: 0});
const files = fs.readFileSync(listPath, 'utf8').split('\n').filter((s) => s.length > 0);
const res = new Float32Array(files.length * ex.dim);
files.forEach((f, i) => {
  const buf = fs.readFileSync(f);
  const samples = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
  const st = ex.createStream();
  st.acceptWaveform({sampleRate: 16000, samples: samples});
  st.inputFinished();
  if (!ex.isReady(st)) throw new Error('not ready: ' + f);
  res.set(ex.compute(st, false), i * ex.dim);
});
fs.writeFileSync(out, Buffer.from(res.buffer));
console.log(JSON.stringify({n: files.length, dim: ex.dim}));
