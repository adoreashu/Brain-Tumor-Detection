import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ProbChart = ({ probabilities }) => {
  const options = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            return ` ${(context.raw * 100).toFixed(2)}%`;
          }
        }
      }
    },
    scales: {
      x: {
        min: 0,
        max: 1,
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.5)',
          callback: (value) => `${value * 100}%`
        }
      },
      y: {
        grid: {
          display: false,
        },
        ticks: {
          color: '#e6f1ff',
          font: {
            family: "'Inter', sans-serif",
            size: 13
          }
        }
      }
    }
  };

  // probabilities is expected to be an object: { 'Glioma': 0.8, 'Meningioma': 0.1, ... }
  const labels = Object.keys(probabilities);
  const dataValues = Object.values(probabilities);

  const data = {
    labels,
    datasets: [
      {
        label: 'Probability',
        data: dataValues,
        backgroundColor: [
          'rgba(0, 212, 170, 0.8)',
          'rgba(0, 180, 216, 0.8)',
          'rgba(255, 107, 107, 0.8)',
          'rgba(255, 204, 0, 0.8)',
        ],
        borderColor: [
          '#00d4aa',
          '#00b4d8',
          '#ff6b6b',
          '#ffcc00',
        ],
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  };

  return (
    <div className="glass-card" style={{ height: '100%' }}>
      <h3 style={{ color: 'var(--color-text-muted)', marginBottom: '1rem', textAlign: 'center' }}>Class Probabilities</h3>
      <div className="chart-container">
        <Bar options={options} data={data} />
      </div>
    </div>
  );
};

export default ProbChart;
