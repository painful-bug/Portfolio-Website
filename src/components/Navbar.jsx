export default function Navbar() {
  return (
    <nav className="navbar">
      {/* Logo */}
      <a href="#" className="navbar__logo group">
        <div className="navbar__logo-circle">
          <span className="navbar__logo-initial">A</span>
        </div>
      </a>

      {/* Desktop Links */}
      <div className="navbar__links">
        <a href="#work" className="navbar__link">WORK</a>
        <a href="#about" className="navbar__link">ABOUT</a>
        <a href="#contact" className="navbar__link">CONTACT</a>
      </div>

      {/* Mobile Menu */}
      <button className="navbar__mobile-btn" aria-label="Open menu">
        <span className="material-symbols-outlined">menu</span>
      </button>
    </nav>
  );
}
