import { useState } from 'react';
import Hero from '../components/Hero';
import UploadZone from '../components/UploadZone';
import ResultCard from '../components/ResultCard';
import ProbChart from '../components/ProbChart';
import GradCamView from '../components/GradCamView';
import InfoSection from '../components/InfoSection';
import Loader from '../components/Loader';
import Modal from '../components/Modal';

const Home = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [activeModal, setActiveModal] = useState(null);

  const handleAnalyze = async (file) => {
    setIsLoading(true);
    setOriginalImage(URL.createObjectURL(file));

    try {
      const formData = new FormData();
      formData.append('file', file);

      const apiUrl = import.meta.env.VITE_API_URL || 
        (window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'https://brain-tumor-detection-3raj.onrender.com');
      const response = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      
      const isTumor = data.prediction.toLowerCase() !== 'notumor';
      let predClass = data.prediction;
      
      if (predClass === 'notumor') predClass = 'No Tumor';
      if (predClass === 'glioma') predClass = 'Glioma';
      if (predClass === 'meningioma') predClass = 'Meningioma';
      if (predClass === 'pituitary') predClass = 'Pituitary Tumor';
      
      const formattedProbs = {
        'Glioma': data.probabilities['glioma'] || 0,
        'Meningioma': data.probabilities['meningioma'] || 0,
        'Pituitary Tumor': data.probabilities['pituitary'] || 0,
        'No Tumor': data.probabilities['notumor'] || 0
      };

      setResult({
        prediction: predClass,
        confidence: data.confidence,
        probabilities: formattedProbs,
        isTumor: isTumor,
        heatmapImage: data.gradcam_image ? `data:image/png;base64,${data.gradcam_image}` : null
      });

    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Failed to analyze image. Please check if the backend server is running.");
    } finally {
      setIsLoading(false);
      setTimeout(() => {
        document.getElementById('results-area')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  };

  return (
    <>
      <Hero 
        onModelsClick={() => setActiveModal('models')} 
        onAccuracyClick={() => setActiveModal('accuracy')} 
      />
      <UploadZone onAnalyze={handleAnalyze} isLoading={isLoading} />
      
      {result && (
        <section id="results-area" className="results-section container">
          <h2 style={{ textAlign: 'center', margin: '3rem 0', fontSize: '2.5rem' }}>
            <span className="gradient-text">Analysis Results</span>
          </h2>
          
          <div className="results-grid">
            <ResultCard 
              prediction={result.prediction} 
              confidence={result.confidence} 
              isTumor={result.isTumor} 
            />
            <ProbChart probabilities={result.probabilities} />
          </div>
          
          {result.isTumor && (
            <GradCamView 
              originalImage={originalImage} 
              heatmapImage={result.heatmapImage} 
            />
          )}
        </section>
      )}
      
      <InfoSection />
      
      {isLoading && <Loader />}

      {/* Interactive Modals */}
      <Modal 
        isOpen={activeModal === 'models'} 
        onClose={() => setActiveModal(null)}
        title="Our 3 ML Models"
      >
        <p>
          We employ a highly advanced <strong>Ensemble Architecture</strong> instead of relying on just a single AI model. This allows us to combine the strengths of different neural network designs for maximum reliability.
        </p>
        
        <h3>1. MobileNetV2</h3>
        <p>
          A lightweight, extremely fast convolutional neural network. It excels at identifying the broad shape and location of tumors within milliseconds.
        </p>

        <h3>2. EfficientNetB0</h3>
        <p>
          A state-of-the-art model that uses compound scaling to extract deep, complex feature maps. It is highly sensitive to subtle differences between Gliomas and Meningiomas.
        </p>

        <h3>3. Ensemble Averaging Logic</h3>
        <p>
          Our third "model" is the ensemble arbitrator. It runs images through both neural networks simultaneously and computes a weighted average of their probability arrays. If one model develops a "blind spot", the other model seamlessly corrects it.
        </p>
      </Modal>

      <Modal 
        isOpen={activeModal === 'accuracy'} 
        onClose={() => setActiveModal(null)}
        title="Achieving 95%+ Accuracy"
      >
        <p>
          Medical AI requires extreme precision. To push our accuracy past the 95% threshold, we implemented several advanced techniques:
        </p>

        <h3>Clean Label Noise Auditing</h3>
        <p>
          Public medical datasets often contain human errors where images are placed in the wrong folder (e.g. a Glioma labeled as a Meningioma). We used AI to audit the dataset and identify 25 corrupted labels, manually correcting them so the model learns from perfect data.
        </p>

        <h3>Transfer Learning</h3>
        <p>
          Instead of training from scratch, our models are initialized with weights learned from millions of real-world images (ImageNet). They are then fine-tuned specifically on MRI scans, allowing them to understand complex textures and edges immediately.
        </p>

        <h3>Data Augmentation</h3>
        <p>
          During training, we rotate, zoom, and adjust the brightness of the MRI scans randomly. This teaches the AI to recognize tumors regardless of how the MRI machine was positioned or calibrated.
        </p>
      </Modal>
    </>
  );
};

export default Home;
