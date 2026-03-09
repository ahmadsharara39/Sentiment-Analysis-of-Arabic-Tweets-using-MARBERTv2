import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

export default function ClassDistributionChart({ predictions }) {
  if (!predictions || predictions.length === 0) return null;

  // Count occurrences of each class
  const labelCounts = predictions.reduce((acc, cur) => {
    acc[cur.label] = (acc[cur.label] || 0) + 1;
    return acc;
  }, {});

  const labels = Object.keys(labelCounts);
  const counts = Object.values(labelCounts);

  const data = {
    labels,
    datasets: [
      {
        label: 'Tweet Count per Class',
        data: counts,
        backgroundColor: '#4CAF50',
      },
    ],
  };

  return (
    <div style={{ marginTop: '2rem' }}>
      <h2 style={{ color: 'white' }}>📊 Class Distribution</h2>
      <Bar data={data} />
    </div>
  );
}
