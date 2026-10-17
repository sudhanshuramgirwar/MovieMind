// Let the cards appear in a small stagger instead of all at once.
document.addEventListener('DOMContentLoaded',()=>{document.querySelectorAll('.movie-card').forEach((card,i)=>{card.style.setProperty('--delay',`${Math.min(i*35,350)}ms`)});});
