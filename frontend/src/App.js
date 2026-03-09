import React, { useState } from 'react';
import ModelSelector from './components/ModelSelector';
import FileUploader from './components/FileUploader';
import SingleTweetInput from './components/SingleTweetInput';
import PredictionsTable from './components/PredictionsTable';
import MetricsDashboard from './components/MetricsDashboard';
import ClassDistributionChart from './components/ClassDistributionChart';
import ConfusionMatrixChart from './components/ConfusionMatrixChart';
import './App.css';

// Simple mapping for numeric labels
const LABEL_MAP = {
  0: 'Negative',
  1: 'Neutral',
  2: 'Positive'
};

function App() {
  const [selectedModel, setSelectedModel] = useState('');
  const [predictions, setPredictions]   = useState([]);
  const [metrics, setMetrics]           = useState(null);

  return (
    <div className="App" style={{ maxWidth: '900px', margin: 'auto', padding: '2rem' }}>
      <h1>🧠 Arabic Tweet Sentiment Analyzer</h1>

      {/* Model Selector */}
      <ModelSelector
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
      />

      {/* Legend for your labels */}
      <div style={{ margin: '1rem 0' }}>
        <strong>Label Mapping:</strong>{' '}
        {Object.entries(LABEL_MAP)
          .map(([num, text]) => `${num} = ${text}`)
          .join(' | ')}
      </div>

      <hr />

      {/* Single Tweet Prediction */}
      <SingleTweetInput
        selectedModel={selectedModel}
        labelMap={LABEL_MAP}
      />

      <hr />

      {/* Bulk CSV Prediction */}
      <FileUploader
        selectedModel={selectedModel}
        setPredictions={setPredictions}
        setMetrics={setMetrics}
      />

      {/* Metrics Display */}
      <MetricsDashboard metrics={metrics} />

      {/* Predictions Table */}
      <PredictionsTable
        predictions={predictions}
        labelMap={LABEL_MAP}
      />

      {/* Class Distribution */}
      <ClassDistributionChart predictions={predictions} />

      {/* Confusion Matrix */}
      {metrics?.confusion_matrix && (
        <ConfusionMatrixChart matrix={metrics.confusion_matrix} />
      )}
    </div>
  );
}

export default App;
