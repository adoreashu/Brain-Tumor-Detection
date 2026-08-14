import React from 'react';

const FeatureExtractionView = ({ originalImage, edgeImage, textureImage, gradcamImage }) => {
  return (
    <div className="glass-card feature-extraction-card" style={{ marginTop: '2rem', padding: '2rem' }}>
      <h3 style={{ textAlign: 'center', color: 'var(--color-primary-light)', marginBottom: '0.5rem' }}>
        Deep Feature Extraction Analysis
      </h3>
      <p style={{ textAlign: 'center', color: 'var(--color-text-muted)', marginBottom: '2rem', maxWidth: '800px', margin: '0 auto 2rem auto' }}>
        This module turns the AI into a "glass box" by mathematically simulating and visualizing the internal thought process of the DenseNet121 neural network as it analyzes the MRI scan.
      </p>

      <div style={{ display: 'flex', flexDirection: 'row', gap: '1.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
        
        {/* Step 1: Shapes & Edges */}
        <div style={{ flex: '1 1 250px', maxWidth: '300px', textAlign: 'center' }}>
          <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.1)', marginBottom: '1rem' }}>
            <img 
              src={edgeImage || originalImage} 
              alt="Shapes and Edges" 
              style={{ width: '100%', display: 'block' }}
            />
          </div>
          <h4 style={{ color: 'var(--color-text)', marginBottom: '0.5rem' }}>Step 1: Shapes & Edges</h4>
          <p style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>
            Simulating early neural network layers. The AI isolates the structural boundaries and geometric shapes of the brain.
          </p>
        </div>

        {/* Step 2: Abnormal Tissue Textures */}
        <div style={{ flex: '1 1 250px', maxWidth: '300px', textAlign: 'center' }}>
          <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.1)', marginBottom: '1rem' }}>
            <img 
              src={textureImage || originalImage} 
              alt="Abnormal Tissue Textures" 
              style={{ width: '100%', display: 'block' }}
            />
          </div>
          <h4 style={{ color: 'var(--color-text)', marginBottom: '0.5rem' }}>Step 2: Tissue Textures</h4>
          <p style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>
            Simulating middle layers. The AI applies high-pass filters to detect abnormal density, bumps, and micro-textures.
          </p>
        </div>

        {/* Step 3: Classification Attention */}
        <div style={{ flex: '1 1 250px', maxWidth: '300px', textAlign: 'center' }}>
          <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.1)', marginBottom: '1rem' }}>
            <img 
              src={gradcamImage || originalImage} 
              alt="Classification Attention" 
              style={{ width: '100%', display: 'block' }}
            />
          </div>
          <h4 style={{ color: 'var(--color-text)', marginBottom: '0.5rem' }}>Step 3: Classification</h4>
          <p style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>
            The final deep layers synthesize all features to predict whether a tumor exists and label its specific category.
          </p>
        </div>

      </div>
    </div>
  );
};

export default FeatureExtractionView;
