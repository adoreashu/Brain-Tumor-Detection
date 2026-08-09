import { useState } from 'react';

const GradCamView = ({ originalImage, heatmapImage }) => {
  const [opacity, setOpacity] = useState(0.5);

  return (
    <div className="glass-card gradcam-view">
      <h3 style={{ textAlign: 'center', marginBottom: '1.5rem', color: 'var(--color-text-muted)' }}>
        Grad-CAM Analysis
      </h3>
      
      <p style={{ textAlign: 'center', fontSize: '0.9rem', marginBottom: '2rem', color: 'var(--color-text-light)' }}>
        Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions in the MRI scan that the AI model focused on most when making its prediction.
      </p>

      <div className="gradcam-images">
        <div style={{ position: 'relative', width: '300px', height: '300px', margin: '0 auto' }}>
          <img 
            src={originalImage} 
            alt="Original MRI" 
            style={{ 
              width: '100%', 
              height: '100%', 
              objectFit: 'cover',
              borderRadius: '8px'
            }} 
          />
          <img 
            src={heatmapImage} 
            alt="Grad-CAM Heatmap" 
            style={{ 
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%', 
              height: '100%', 
              objectFit: 'cover',
              opacity: opacity,
              borderRadius: '8px',
              mixBlendMode: 'screen'
            }} 
          />
        </div>
      </div>

      <div className="opacity-slider-container" style={{ maxWidth: '400px', margin: '0 auto' }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>MRI</span>
        <input 
          type="range" 
          min="0" 
          max="1" 
          step="0.05" 
          value={opacity} 
          onChange={(e) => setOpacity(parseFloat(e.target.value))} 
        />
        <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>Heatmap</span>
      </div>
    </div>
  );
};

export default GradCamView;
