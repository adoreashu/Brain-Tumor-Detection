import { ChevronDown } from 'lucide-react';

const Hero = ({ onModelsClick, onAccuracyClick }) => {
  return (
    <section className="hero">
      <div className="container hero-content">
        <h1>
          <span className="gradient-text">Early Brain Tumor Detections in Human Brain</span>
        </h1>
        <p>AI-Powered MRI Analysis Using Deep Learning. Upload a brain MRI scan to instantly detect the presence and class of brain tumors with high accuracy.</p>
        
        <div className="hero-stats">
          <button className="clickable-stat" onClick={onModelsClick}>
            3 ML Models
          </button>
          <span className="stat-divider">•</span>
          <span className="clickable-stat" style={{ cursor: 'default' }}>
            4 Tumor Classes
          </span>
          <span className="stat-divider">•</span>
          <button className="clickable-stat" onClick={onAccuracyClick}>
            95%+ Accuracy
          </button>
        </div>
      </div>
      
      <div className="scroll-indicator">
        <ChevronDown size={32} />
      </div>
    </section>
  );
};

export default Hero;
