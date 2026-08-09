import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, X, Activity } from 'lucide-react';

const UploadZone = ({ onAnalyze, isLoading }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles?.length > 0) {
      const selectedFile = acceptedFiles[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png']
    },
    maxFiles: 1,
    multiple: false
  });

  const handleClear = (e) => {
    e.stopPropagation();
    setFile(null);
    setPreview(null);
  };

  const handleAnalyze = () => {
    if (file) {
      onAnalyze(file);
    }
  };

  return (
    <section className="upload-section">
      <div className="container">
        <div className="glass-card">
          <h2 style={{ textAlign: 'center', marginBottom: '2rem' }}>Upload MRI Scan</h2>
          
          {!preview ? (
            <div 
              {...getRootProps()} 
              className={`dropzone-container ${isDragActive ? 'active' : ''}`}
            >
              <input {...getInputProps()} />
              <UploadCloud className="upload-icon" />
              <p className="upload-text">
                {isDragActive ? 'Drop the MRI image here' : 'Drag & drop MRI scan here'}
              </p>
              <p className="upload-hint">Supported formats: JPEG, JPG, PNG</p>
              
              <button className="btn-secondary" style={{ marginTop: '1.5rem' }}>
                Browse Files
              </button>
            </div>
          ) : (
            <div className="preview-container">
              <div style={{ position: 'relative' }}>
                <img src={preview} alt="MRI Preview" className="preview-image" />
                <button 
                  onClick={handleClear}
                  style={{
                    position: 'absolute',
                    top: '-10px',
                    right: '-10px',
                    background: 'var(--color-error)',
                    color: 'white',
                    borderRadius: '50%',
                    padding: '0.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 4px 10px rgba(0,0,0,0.3)'
                  }}
                >
                  <X size={20} />
                </button>
              </div>
              
              <p style={{ color: 'var(--color-text-muted)' }}>
                {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
              </p>
              
              <button 
                className="btn-primary" 
                onClick={handleAnalyze}
                disabled={isLoading}
              >
                <Activity style={{ marginRight: '0.5rem' }} />
                {isLoading ? 'Analyzing...' : 'Analyze Image'}
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default UploadZone;
