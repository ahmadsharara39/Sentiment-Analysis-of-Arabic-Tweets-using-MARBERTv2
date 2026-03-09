import React from 'react';

export default function MetricsDashboard({ metrics }) {
  if (!metrics) return null;

  return (
    <div style={{ marginTop: '2rem' }}>
      <h2>📊 Model Evaluation Metrics</h2>
      <ul>
        <li><strong>Accuracy:</strong> {(metrics.accuracy * 100).toFixed(2)}%</li>
        <li><strong>F1 Score (Weighted):</strong> {(metrics.f1_score * 100).toFixed(2)}%</li>
      </ul>
    </div>
  );
}
