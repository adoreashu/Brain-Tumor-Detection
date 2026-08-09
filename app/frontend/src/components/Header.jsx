import { Link, useLocation } from 'react-router-dom';
import { Brain } from 'lucide-react';

const Header = () => {
  const location = useLocation();

  return (
    <header className="header">
      <div className="container">
        <Link to="/" className="logo-container">
          <Brain className="logo-icon" color="#00d4aa" />
          <span>Brain Tumour Detection System by Ashu</span>
        </Link>
        <nav className="nav-links">
          <Link 
            to="/" 
            className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
          >
            Home
          </Link>
          <Link 
            to="/about" 
            className={`nav-link ${location.pathname === '/about' ? 'active' : ''}`}
          >
            About
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
