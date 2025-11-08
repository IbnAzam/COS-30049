// src/pages/Stats.jsx
import { StatsPage } from '../styles/Stats.styled';
import ConfidenceDonut from '../dataVis/ConfidenceDonut';

function Stats() {
  // mock values for demonstration; later, these could come from your backend or local state
  const spamRate = 0.82;
  const hamRate = 1 - spamRate;

  return (
    <StatsPage>
      <h1>Stats</h1>
      <p style={{ color: '#666' }}>Visual breakdown of classification confidence</p>

      <div style={{
        display: 'flex',
        gap: '40px',
        justifyContent: 'center',
        alignItems: 'center',
        marginTop: '30px',
        flexWrap: 'wrap'
      }}>
        <ConfidenceDonut label="Spam" probability={spamRate} />
        <ConfidenceDonut label="Ham" probability={hamRate} />
      </div>
    </StatsPage>
  );
}

export default Stats;
