import { StatsPage } from '../styles/Stats.styled';
import StackedBar7d from '../dataVis/StackedBar7d';

export default function Stats() {
  return (
    <StatsPage>
      <h1>Stats</h1>

      <div className="fullwidth">
        <h3>Spam vs Ham (Last 7 Days)</h3>
        <StackedBar7d width={1200} height={250} />
      </div>

      
    </StatsPage>
  );
}
