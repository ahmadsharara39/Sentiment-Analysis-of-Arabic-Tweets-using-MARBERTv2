import React, { useState } from 'react';
import api from '../api';

export default function FileUploader({ selectedModel, setPredictions, setMetrics }) {
  const [file, setFile] = useState(null);

  const handleUpload = () => {
    const formData = new FormData();
    formData.append('model', selectedModel);
    formData.append('file', file);

    api.post('/predict', formData).then(res => {
      setPredictions(res.data.predictions);
      setMetrics(res.data.metrics);
    });
  };

  return (
    <div>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleUpload} disabled={!file || !selectedModel}>
        Predict CSV
      </button>
    </div>
  );
}
