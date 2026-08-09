import { ChevronDown } from 'lucide-react';

const Hero = () => {
  return (
    <section className="hero">
      <div className="container hero-content">
        <h1>
          <span className="gradient-text">Brain Tumor Detection</span>
        </h1>
        <p>AI-Powered MRI Analysis Using Deep Learning. Upload a brain MRI scan to instantly detect the presence and class of brain tumors with high accuracy.</p>
        
        <div className="hero-stats">
          <span>3 ML Models</span>
          <span>•</span>
          <span>4 Tumor Classes</span>
          <span>•</span>
          <span>95%+ Accuracy</span>
        </div>
      </div>
      
      <div className="scroll-indicator">
        <ChevronDown size={32} />
      </div>
    </section>
  );
};

export default Hero;
