// Run one ONNX speaker model with onnxruntime-node on a float32 feature file (1, T, 80).
// Used by 13_ort_version_check.py; install a version with:
//   npm install --ignore-scripts onnxruntime-node@<version>   (in a separate folder)
// usage: node run.js <ort module dir> <model.onnx> <feats.f32> <T> <out.f32>
const [modDir, model, featsPath, T, out] = process.argv.slice(2);
const ort = require(modDir);
const fs = require('fs');
(async () => {
  const buf = fs.readFileSync(featsPath);
  const data = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  const s = await ort.InferenceSession.create(model, { intraOpNumThreads: 1 });
  const r = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', data, [1, parseInt(T), 80]) });
  fs.writeFileSync(out, Buffer.from(r[s.outputNames[0]].data.buffer));
})();
