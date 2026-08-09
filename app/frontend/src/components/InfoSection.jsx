import { Activity, ShieldCheck, AlertCircle, AlertTriangle } from 'lucide-react';

const InfoSection = () => {
  const cards = [
    {
      title: "Glioma",
      description: "A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the gluey supportive cells (glial cells) that surround nerve cells and help them function.",
      icon: <AlertTriangle color="#ffcc00" size={32} />
    },
    {
      title: "Meningioma",
      description: "A tumor that arises from the meninges — the membranes that surround your brain and spinal cord. Most are noncancerous (benign), though rarely, a meningioma may be malignant.",
      icon: <AlertCircle color="#ff6b6b" size={32} />
    },
    {
      title: "Pituitary Tumor",
      description: "Abnormal growths that develop in your pituitary gland. Some pituitary tumors result in too many of the hormones that regulate important functions of your body.",
      icon: <Activity color="#00b4d8" size={32} />
    },
    {
      title: "No Tumor",
      description: "A healthy brain scan with no signs of abnormal growths, masses, or tumors detected by the deep learning model.",
      icon: <ShieldCheck color="#00d4aa" size={32} />
    }
  ];

  return (
    <section className="info-section">
      <div className="container">
        <h2>Understanding Brain Tumors</h2>
        <div className="info-grid">
          {cards.map((card, idx) => (
            <div key={idx} className="glass-card info-card">
              <div className="info-icon">{card.icon}</div>
              <h3>{card.title}</h3>
              <p style={{ color: 'var(--color-text-muted)', fontSize: '0.95rem' }}>
                {card.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default InfoSection;
