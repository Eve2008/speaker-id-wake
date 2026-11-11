// Run one ONNX speaker model with onnxruntime-node on many float32 feature files.
// usage: node ort_node_batch.js <ort module dir> <model.onnx> <list.txt>
// list.txt: one line per utterance: "<feats.f32> <T> <out.f32>" (features shaped 1 x T x 80)
const [modDir, model, listPath] = process.argv.slice(2);
const ort = require(modDir);
const fs = require('fs');
(async () => {
  const s = await ort.InferenceSession.create(model, { intraOpNumThreads: 1, interOpNumThreads: 1 });
  const lines = fs.readFileSync(listPath, 'utf8').split('\n').filter((l) => l.trim());
  for (const line of lines) {
    const [fp, T, out] = line.trim().split(/\s+/);
    const buf = fs.readFileSync(fp);
    const data = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
    const r = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', data, [1, parseInt(T), 80]) });
    fs.writeFileSync(out, Buffer.from(r[s.outputNames[0]].data.buffer));
  }
})();
