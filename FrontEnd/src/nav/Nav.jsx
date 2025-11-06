// src/nav/Nav.jsx
import { NavLink } from 'react-router-dom';

function Nav() {
  return (
    <nav>
      <ul>
        <li><NavLink to="/" end>Home</NavLink></li>
        <li><NavLink to="/scan">Scan</NavLink></li>
        <li><NavLink to="/stats">Stats</NavLink></li>
      </ul>
    </nav>
  );
}
export default Nav;
