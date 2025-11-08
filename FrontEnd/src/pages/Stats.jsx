import { StatsPage } from '../styles/Stats.styled';
import StackedBar7d from '../dataVis/StackedBar7d';
import PieSpamHam from "../dataVis/PieSpamHam";
import ProbabilityHistogram from "../dataVis/ProbabilityHistogram";


export default function Stats() {
  return (
    <StatsPage>
      <h1>Stats</h1>

      {/* Top row: full-width stacked bars */}
      <section className="card fullwidth">
        <h4 style={{ marginTop: 0 }}>Spam vs Ham</h4>
        <StackedBar7d height={250} width={1100} />
      </section>

      {/* Bottom row: two cards side-by-side */}
      <section className="card">
        <h4 style={{ marginTop: 0 }}>Total Ratio</h4>
        <PieSpamHam height={350} width={350} />
      </section>

      <section className="card span-2">
        <h4 style={{ marginTop: 0 }}>Confidence</h4>
        <ProbabilityHistogram height={350} width={800} bins={20} />
      </section>
    </StatsPage>
  );
}