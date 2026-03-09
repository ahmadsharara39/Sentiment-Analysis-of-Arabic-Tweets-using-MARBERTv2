import React, { useState } from 'react';
import api from '../api';

// Accepting labelMap as a prop
export default function SingleTweetInput({ selectedModel, labelMap }) {
  const [tweet, setTweet] = useState('');
  const [result, setResult] = useState(null);

  const handleSinglePredict = () => {
    api.post('/predict-single', { tweet, model: selectedModel })
      .then(res => setResult(res.data))
      .catch(err => {
        console.error(err);
        alert('Prediction failed.');
      });
  };

  const labelText = result ? (labelMap[result.label] || 'Unknown') : ''; // Map numeric label to text

  return (
    <div>
      <textarea
        value={tweet}
        onChange={e => setTweet(e.target.value)}
        placeholder="Enter tweet..."
      />
      <button
        onClick={handleSinglePredict}
        disabled={!tweet || !selectedModel}
      >
        Predict Tweet
      </button>

      {result && (
        <div style={{ marginTop: '1rem' }}>
          <h3>
            Sentiment: {result.label} — {labelText}
          </h3>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}
