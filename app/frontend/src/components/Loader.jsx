import { useEffect, useState } from 'react';

const Loader = ({ message = "Analyzing MRI scan..." }) => {
  const [dots, setDots] = useState('');

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="loader-overlay">
      <div className="scan-animation">
        <div className="scan-line"></div>
      </div>
      <div className="loader-text">
        {message}{dots}
      </div>
      <p style={{ marginTop: '1rem', color: 'var(--color-text-muted)', fontSize: '0.9rem' }}>
        Running deep learning models. This may take a few seconds.
      </p>
    </div>
  );
};

export default Loader;
