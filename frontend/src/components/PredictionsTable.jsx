import React from 'react';

// Accepting labelMap as a prop
export default function PredictionsTable({ predictions, labelMap }) {
  if (!predictions || predictions.length === 0) return null;

  return (
    <div style={{ marginTop: '2rem' }}>
      <h2 style={{ color: 'white' }}>📄 Prediction Results</h2>
      <table border="1" cellPadding="8">
        <thead>
          <tr>
            <th>#</th>
            <th>Tweet</th>
            <th>Predicted Label</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((row, idx) => {
            const labelText = labelMap[row.label] || 'Unknown'; // Map numeric label to text
            return (
              <tr key={idx}>
                <td>{idx + 1}</td>
                <td>{row.text}</td>
                <td>
                  {row.label} — {labelText}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
