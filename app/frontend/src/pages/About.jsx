import { Github, Database, BrainCircuit, Activity } from 'lucide-react';

const About = () => {
  return (
    <div className="container" style={{ padding: '4rem 0' }}>
      <div className="glass-card" style={{ maxWidth: '900px', margin: '0 auto' }}>
        <h1 className="gradient-text" style={{ fontSize: '3rem', marginBottom: '2rem', textAlign: 'center' }}>
          About Brain Tumour Detection System by Ashu
        </h1>
        
        <section style={{ marginBottom: '3rem' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: 'var(--color-secondary)' }}>
            <BrainCircuit />
            The Project
          </h2>
          <p style={{ color: 'var(--color-text-muted)', fontSize: '1.1rem', marginBottom: '1rem' }}>
            Brain Tumour Detection System by Ashu is a deep learning-powered web application designed to assist in the rapid detection and classification of brain tumors from MRI scans. Using a state-of-the-art Convolutional Neural Network (CNN), the system can classify scans into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.
          </p>
          <p style={{ color: 'var(--color-text-muted)', fontSize: '1.1rem' }}>
            The frontend provides an intuitive, premium interface for medical professionals and researchers to upload scans, view prediction confidences, and analyze model attention through Grad-CAM visualizations.
          </p>
        </section>

        <section style={{ marginBottom: '3rem' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: 'var(--color-secondary)' }}>
            <Database />
            Dataset & Training
          </h2>
          <ul style={{ color: 'var(--color-text-muted)', fontSize: '1.1rem', paddingLeft: '2rem', lineHeight: '1.8' }}>
            <li>Trained on a massive clinical Mega-Dataset of over 13,100 Brain MRI Images.</li>
            <li>Images preprocessed using standard normalization and resizing techniques.</li>
            <li>Achieved a robust 95% accuracy on an unseen test set of 3,381 images.</li>
            <li>Uses data augmentation to ensure model robustness.</li>
          </ul>
        </section>

        <section style={{ marginBottom: '3rem' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: 'var(--color-secondary)' }}>
            <Activity />
            Technology Stack
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', color: 'var(--color-text-muted)' }}>
            <div className="glass-card" style={{ background: 'rgba(0,0,0,0.2)', padding: '1.5rem' }}>
              <h3 style={{ color: 'var(--color-primary)', marginBottom: '1rem' }}>Frontend</h3>
              <ul>
                <li>React 18</li>
                <li>Vite</li>
                <li>Chart.js</li>
                <li>CSS Modules / Custom Styling</li>
              </ul>
            </div>
            <div className="glass-card" style={{ background: 'rgba(0,0,0,0.2)', padding: '1.5rem' }}>
              <h3 style={{ color: 'var(--color-primary)', marginBottom: '1rem' }}>Backend / AI</h3>
              <ul>
                <li>Python & FastAPI</li>
                <li>ONNX Runtime (DenseNet121)</li>
                <li>OpenCV</li>
                <li>NumPy (Grad-CAM)</li>
              </ul>
            </div>
          </div>
        </section>

        <div style={{ textAlign: 'center', marginTop: '4rem', padding: '2rem', borderTop: '1px solid var(--color-border)' }}>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="btn-secondary" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
            <Github size={20} />
            View Source on GitHub
          </a>
        </div>
      </div>
    </div>
  );
};

export default About;
