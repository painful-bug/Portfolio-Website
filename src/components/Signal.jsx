import { useState } from 'react';

export default function Signal() {
  const [formStatus, setFormStatus] = useState('idle'); // idle | sending | sent | error

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormStatus('sending');

    const form = e.target;
    const data = new FormData(form);

    try {
      const res = await fetch('https://formsubmit.co/ajax/ujanaishik109@gmail.com', {
        method: 'POST',
        headers: { 'Accept': 'application/json' },
        body: data,
      });

      if (res.ok) {
        setFormStatus('sent');
        form.reset();
        setTimeout(() => setFormStatus('idle'), 4000);
      } else {
        setFormStatus('error');
        setTimeout(() => setFormStatus('idle'), 4000);
      }
    } catch {
      setFormStatus('error');
      setTimeout(() => setFormStatus('idle'), 4000);
    }
  };

  return (
    <footer className="signal" id="contact">
      {/* Contact Section */}
      <div className="signal__contact-section">
        <div className="signal__contact-grid">
          {/* Left: CTA Text */}
          <div className="signal__contact-left">
            <div className="signal__label">
              <div className="signal__label-line" />
              <span className="signal__label-text">Get in Touch</span>
            </div>
            <h2 className="signal__contact-heading font-serif">
              Let&apos;s build something<br />
              <span className="signal__contact-accent">together.</span>
            </h2>
            <p className="signal__contact-desc">
              Have a project in mind, a research collaboration, or just want to
              say hello? Drop me a message and I&apos;ll get back to you.
            </p>

            {/* Direct Links */}
            <div className="signal__direct-links">
              <a href="mailto:ujanaishik109@gmail.com" className="signal__direct-link">
                <span className="material-symbols-outlined signal__direct-icon">mail</span>
                <span>ujanaishik109@gmail.com</span>
              </a>
              <a href="https://linkedin.com/in/aishik-bandyopadhyay" className="signal__direct-link" target="_blank" rel="noopener noreferrer">
                <span className="material-symbols-outlined signal__direct-icon">link</span>
                <span>LinkedIn</span>
              </a>
              <a href="https://github.com/aishik-b" className="signal__direct-link" target="_blank" rel="noopener noreferrer">
                <span className="material-symbols-outlined signal__direct-icon">code</span>
                <span>GitHub</span>
              </a>
            </div>
          </div>

          {/* Right: Contact Form */}
          <div className="signal__contact-right">
            <form
              className="signal__form glass-card"
              onSubmit={handleSubmit}
            >
              {/* Honeypot for spam prevention */}
              <input type="text" name="_honey" style={{ display: 'none' }} />
              {/* Disable captcha page */}
              <input type="hidden" name="_captcha" value="false" />
              <input type="hidden" name="_subject" value="New message from Portfolio" />

              <div className="signal__form-group">
                <label className="signal__form-label" htmlFor="contact-name">Name</label>
                <input
                  className="signal__form-input"
                  type="text"
                  id="contact-name"
                  name="name"
                  placeholder="Your name"
                  required
                />
              </div>

              <div className="signal__form-group">
                <label className="signal__form-label" htmlFor="contact-email">Email</label>
                <input
                  className="signal__form-input"
                  type="email"
                  id="contact-email"
                  name="email"
                  placeholder="you@example.com"
                  required
                />
              </div>

              <div className="signal__form-group">
                <label className="signal__form-label" htmlFor="contact-message">Message</label>
                <textarea
                  className="signal__form-textarea"
                  id="contact-message"
                  name="message"
                  placeholder="Tell me about your project, idea, or just say hi..."
                  rows="5"
                  required
                />
              </div>

              <button
                type="submit"
                className="signal__form-submit"
                disabled={formStatus === 'sending'}
              >
                {formStatus === 'idle' && (
                  <>
                    <span>Send Message</span>
                    <span className="material-symbols-outlined signal__form-submit-icon">arrow_forward</span>
                  </>
                )}
                {formStatus === 'sending' && <span>Sending...</span>}
                {formStatus === 'sent' && (
                  <>
                    <span className="material-symbols-outlined" style={{ fontSize: '1.125rem' }}>check_circle</span>
                    <span>Message Sent!</span>
                  </>
                )}
                {formStatus === 'error' && <span>Something went wrong. Try again.</span>}
              </button>
            </form>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="signal__bottom">
        <div className="signal__bottom-inner">
          <span className="signal__copy">© 2025 Aishik Bandyopadhyay</span>
          <div className="signal__bottom-links">
            <a href="mailto:ujanaishik109@gmail.com" className="signal__bottom-link">Email</a>
            <a href="https://linkedin.com/in/aishik-bandyopadhyay" className="signal__bottom-link" target="_blank" rel="noopener noreferrer">LinkedIn</a>
            <a href="https://github.com/aishik-b" className="signal__bottom-link" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
