import React, { useEffect, useState } from 'react';
import api from '../api';

export default function ModelSelector({ selectedModel, setSelectedModel }) {
  const [models, setModels] = useState([]);

  useEffect(() => {
    api.get('/models').then(res => {
      // Remove 'ensemble' from the models list
      setModels(res.data.filter(model => model !== 'ensemble'));
    });
  }, []);

  return (
    <div>
      <label htmlFor="model">Select Model: </label>
      <select
        id="model"
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
      >
        <option value="" disabled>Choose a model</option> {/* "Choose a model" as static text */}
        {models.map(model => (
          <option key={model} value={model}>{model}</option>
        ))}
      </select>

      {/* Display selected model as a label */}
      {selectedModel && (
        <div style={{ marginTop: '10px', fontSize: '1.2rem', color: '#42a5f5' }}>
          Selected Model: <strong>{selectedModel}</strong>
        </div>
      )}
    </div>
  );
}
