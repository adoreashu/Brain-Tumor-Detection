import { useState } from 'react';
import Hero from '../components/Hero';
import UploadZone from '../components/UploadZone';
import ResultCard from '../components/ResultCard';
import ProbChart from '../components/ProbChart';
import GradCamView from '../components/GradCamView';
import InfoSection from '../components/InfoSection';
import Loader from '../components/Loader';

const Home = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);

  const handleAnalyze = async (file) => {
    setIsLoading(true);
    setOriginalImage(URL.createObjectURL(file));

    try {
      const formData = new FormData();
      formData.append('file', file);

      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
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
      <Hero />
      <UploadZone onAnalyze={handleAnalyze} isLoading={isLoading} />
      
      {result && (
        <section id="results-area" className="results-section container">
          <h2 style={{ textAlign: 'center', marginBottom: '3rem', fontSize: '2.5rem' }}>
            Analysis Results
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
    </>
  );
};

export default Home;
