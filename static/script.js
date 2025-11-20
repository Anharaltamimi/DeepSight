document.addEventListener('DOMContentLoaded', () => {
  // ===== إدارة قائمة الأفاتار =====
  const userIcon = document.getElementById('userIcon');
  const dropdownMenu = document.getElementById('dropdownMenu');

  if (userIcon && dropdownMenu) {
    userIcon.addEventListener('click', e => {
      dropdownMenu.classList.toggle('show');
      e.stopPropagation();
    });

    dropdownMenu.addEventListener('click', e => e.stopPropagation());

    document.addEventListener('click', () => dropdownMenu.classList.remove('show'));

    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') dropdownMenu.classList.remove('show');
    });
  }

  // (اختياري) تحديث اسم المستخدم والرمز
  const nameEl = document.getElementById('username');
  const stored = localStorage.getItem('userID');
  if (nameEl && stored && stored.trim()) {
    nameEl.textContent = stored.trim();
    if (userIcon) userIcon.textContent = stored.trim().charAt(0).toUpperCase();
  }

  // ===== إدارة الـ Active للروابط والتنقل بين الأقسام =====
  const links = document.querySelectorAll('.nav-left .nav-link');

  function setActive(target) {
    links.forEach(link => link.classList.remove('active'));
    const activeLink = Array.from(links).find(l => l.dataset.target === target);
    if (activeLink) activeLink.classList.add('active');
  }

  // ✅ عند بداية الصفحة: فعّل Home
  setActive('home');

  // ✅ عند الضغط على الروابط
  links.forEach(link => {
    link.addEventListener('click', e => {
      const target = link.dataset.target;

      if (['about', 'team', 'feature'].includes(target)) {
        e.preventDefault();
        const section = document.getElementById(target);
        if (section) {
          section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }

      setActive(target);
    });
  });

  // ✅ تحديث Active عند Scroll
  window.addEventListener('scroll', () => {
    const sections = ['feature', 'about', 'team'];

    const scrollPos = window.scrollY;
    let activeSection = 'home';

    // إذا المستخدم فوق أول قسم → Home active
    const firstSection = document.getElementById(sections[0]);
    if (firstSection && scrollPos < firstSection.offsetTop - 150) {
      setActive('home');
      return;
    }

    // تحديد القسم النشط
    for (let id of sections) {
      const el = document.getElementById(id);
      if (!el) continue;

      const top = el.offsetTop - 200;
      const bottom = top + el.offsetHeight;

      if (scrollPos >= top && scrollPos < bottom) {
        activeSection = id;
        break;
      }
    }

    setActive(activeSection);
  });
});
