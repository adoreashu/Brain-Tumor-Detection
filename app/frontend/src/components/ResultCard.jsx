import React from 'react';

const ResultCard = ({ prediction, confidence, isTumor, tumorPercentage }) => {
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
      
      {isTumor && tumorPercentage !== undefined && tumorPercentage !== null && (
        <div style={{
          marginTop: '1.5rem',
          padding: '0.75rem 1rem',
          backgroundColor: 'rgba(255, 60, 60, 0.1)',
          border: '1px solid rgba(255, 60, 60, 0.3)',
          borderRadius: '8px',
          textAlign: 'center'
        }}>
          <h4 style={{ color: '#ff4d4d', margin: 0, fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Affected Brain Area
          </h4>
          <p style={{ color: 'white', margin: '0.5rem 0 0 0', fontSize: '1.5rem', fontWeight: 'bold' }}>
            ~{tumorPercentage}%
          </p>
        </div>
      )}
    </div>
  );
};

export default ResultCard;
