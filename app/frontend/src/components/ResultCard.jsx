import React from 'react';

const ResultCard = ({ prediction, confidence, isTumor }) => {
  // Convert confidence to a percentage for CSS conic gradient
  const confidencePercent = (confidence * 100).toFixed(1);
  const progressStyle = { '--progress': `${confidence * 100}%` };

  return (
    <div className="glass-card result-card" style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
      <div className="result-header">
        <h3 style={{ color: 'var(--color-text-muted)', marginBottom: '1.5rem' }}>Analysis Result</h3>
      </div>
      
      <div className="confidence-circle" style={progressStyle}>
        <span className="confidence-value">{confidencePercent}%</span>
      </div>
      
      <h2 className={`prediction-label ${isTumor ? 'tumor' : 'normal'}`}>
        {prediction}
      </h2>
      
      <div className={`severity-badge ${isTumor ? 'tumor' : 'normal'}`}>
        {isTumor ? 'Tumor Detected' : 'No Tumor Detected'}
      </div>
    </div>
  );
};

export default ResultCard;
