import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend
);

export default function ConfusionMatrixChart({ matrix }) {
  if (!matrix) return null;

  // e.g. matrix = [[10,1,0], [2,15,1], [0,1,9]]
  const classLabels = matrix.map((_, i) => `Class ${i}`);

  const data = {
    labels: classLabels,             // Actual class
    datasets: matrix.map((row, i) => ({
      label: `Predicted ${i}`,        // Predicted class
      data: row,                      // Counts for each actual class
      // backgroundColor omitted → uses Chart.js defaults
    })),
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: '🔲 Confusion Matrix' },
    },
    scales: {
      x: {
        title: { display: true, text: 'Actual Class' },
      },
      y: {
        title: { display: true, text: 'Count' },
        beginAtZero: true,
      },
    },
  };

  return (
    <div style={{ marginTop: '2rem' }}>
      <Bar data={data} options={options} />
    </div>
  );
}
