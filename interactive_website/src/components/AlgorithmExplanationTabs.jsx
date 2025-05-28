import React, { useState } from 'react';
import '../styles.css';

/**
 * AlgorithmExplanationTabs component for displaying educational content
 * about each optimization algorithm.
 */
const AlgorithmExplanationTabs = ({ selectedAlgorithm }) => {
  const [activeTab, setActiveTab] = useState('overview');
  
  // Define tabs for each algorithm
  const tabs = {
    overview: 'Overview',
    operators: 'Operators',
    parameters: 'Parameters',
    metrics: 'Metrics',
    visualization: 'Visualization Guide'
  };
  
  // Algorithm-specific content
  const algorithmContent = {
    HillClimbing: {
      overview: (
        <div className="algorithm-explanation">
          <h3>Hill Climbing Algorithm</h3>
          <p>
            Hill Climbing is a local search optimization algorithm that starts with an arbitrary solution
            and iteratively makes small improvements to reach a better solution. It's analogous to a hiker
            climbing a hill and always moving in the direction of steepest ascent.
          </p>
          <p>
            In the Fantasy League context, Hill Climbing works by:
          </p>
          <ol>
            <li>Starting with a random team assignment</li>
            <li>Exploring neighboring solutions by swapping players between teams</li>
            <li>Accepting moves that improve the fitness (reduce skill standard deviation)</li>
            <li>Continuing until no better neighboring solution can be found</li>
          </ol>
          <p>
            <strong>Strengths:</strong> Simple to implement, requires little memory, and can quickly find
            good solutions for simple problems.
          </p>
          <p>
            <strong>Weaknesses:</strong> Prone to getting stuck in local optima, especially in complex
            search spaces like the Fantasy League problem.
          </p>
        </div>
      ),
      operators: (
        <div className="algorithm-explanation">
          <h3>Hill Climbing Operators</h3>
          <p>
            Hill Climbing uses the following operators to navigate the search space:
          </p>
          <div className="operator-card">
            <h4>Swap Mutation</h4>
            <p>
              The primary operator in Hill Climbing is the swap mutation, which exchanges players
              between teams to create neighboring solutions.
            </p>
            <p>
              <strong>How it works:</strong> Two teams are randomly selected, and one player from
              each team is swapped. The new solution is evaluated, and if it improves the fitness,
              the swap is accepted.
            </p>
            <div className="operator-visualization">
              <div className="team-before">
                <div className="team-label">Team A</div>
                <div className="player-list">
                  <div className="player">Player 1</div>
                  <div className="player highlighted">Player 2</div>
                  <div className="player">Player 3</div>
                </div>
              </div>
              <div className="swap-arrow">↔</div>
              <div className="team-before">
                <div className="team-label">Team B</div>
                <div className="player-list">
                  <div className="player">Player 4</div>
                  <div className="player highlighted">Player 5</div>
                  <div className="player">Player 6</div>
                </div>
              </div>
            </div>
          </div>
          <div className="operator-card">
            <h4>Random Restart (Optional)</h4>
            <p>
              To escape local optima, Hill Climbing can be enhanced with random restarts.
            </p>
            <p>
              <strong>How it works:</strong> After reaching a local optimum, the algorithm
              restarts from a new random solution. This helps explore different regions of
              the search space.
            </p>
          </div>
          <div className="operator-card">
            <h4>Intensive Local Search (Optional)</h4>
            <p>
              Another enhancement is intensive local search, which explores more neighbors
              before making a decision.
            </p>
            <p>
              <strong>How it works:</strong> Instead of accepting the first improvement,
              the algorithm evaluates multiple neighboring solutions and selects the best one.
            </p>
          </div>
        </div>
      ),
      parameters: (
        <div className="algorithm-explanation">
          <h3>Hill Climbing Parameters</h3>
          <p>
            The following parameters control the behavior of the Hill Climbing algorithm:
          </p>
          <div className="parameter-card">
            <h4>Max Iterations</h4>
            <p>
              The maximum number of iterations the algorithm will perform before stopping.
            </p>
            <p>
              <strong>Impact:</strong> Higher values allow more time for exploration but increase
              computational cost. Lower values may result in premature termination.
            </p>
            <p>
              <strong>Recommended range:</strong> 100-10,000 depending on problem complexity.
            </p>
          </div>
          <div className="parameter-card">
            <h4>Random Restarts</h4>
            <p>
              The number of times the algorithm will restart from a new random solution after
              reaching a local optimum.
            </p>
            <p>
              <strong>Impact:</strong> Higher values increase the chance of finding the global
              optimum but multiply the computational cost.
            </p>
            <p>
              <strong>Recommended range:</strong> 0-20 depending on problem complexity.
            </p>
          </div>
          <div className="parameter-card">
            <h4>Intensive Search</h4>
            <p>
              Whether to use intensive local search, exploring more neighbors before making a decision.
            </p>
            <p>
              <strong>Impact:</strong> When enabled, the algorithm makes more informed decisions
              but requires more evaluations per iteration.
            </p>
            <p>
              <strong>Recommended setting:</strong> Enable for complex problems with many local optima.
            </p>
          </div>
        </div>
      ),
      metrics: (
        <div className="algorithm-explanation">
          <h3>Hill Climbing Metrics</h3>
          <p>
            The following metrics help evaluate the performance of Hill Climbing:
          </p>
          <div className="metric-card">
            <h4>Fitness Value</h4>
            <p>
              The objective function value (skill standard deviation) that we're trying to minimize.
            </p>
            <p>
              <strong>Interpretation:</strong> Lower values indicate more balanced teams.
            </p>
          </div>
          <div className="metric-card">
            <h4>Convergence Rate</h4>
            <p>
              How quickly the algorithm approaches its final solution.
            </p>
            <p>
              <strong>Interpretation:</strong> Steeper curves indicate faster convergence.
              Plateaus suggest the algorithm is stuck in a local optimum.
            </p>
          </div>
          <div className="metric-card">
            <h4>Plateau Detection</h4>
            <p>
              Identifies periods where the algorithm makes no progress.
            </p>
            <p>
              <strong>Interpretation:</strong> Frequent or long plateaus suggest the algorithm
              is struggling with local optima.
            </p>
          </div>
          <div className="metric-card">
            <h4>Local Optima Count</h4>
            <p>
              The number of local optima encountered during the search.
            </p>
            <p>
              <strong>Interpretation:</strong> Higher values indicate a more rugged fitness landscape.
            </p>
          </div>
          <div className="metric-card">
            <h4>Improvement Rate</h4>
            <p>
              The percentage of iterations that result in fitness improvements.
            </p>
            <p>
              <strong>Interpretation:</strong> Higher values indicate more efficient search.
            </p>
          </div>
        </div>
      ),
      visualization: (
        <div className="algorithm-explanation">
          <h3>Hill Climbing Visualization Guide</h3>
          <p>
            The visualization shows how Hill Climbing optimizes team assignments:
          </p>
          <div className="visualization-guide">
            <h4>Team Flags</h4>
            <p>
              Each colored flag represents a team. Players are shown as cards beneath their team flag.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Player Movements</h4>
            <p>
              When the algorithm swaps players between teams, you'll see the player cards move
              from one team to another. These movements represent the algorithm exploring neighboring
              solutions.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Fitness Display</h4>
            <p>
              The current and best fitness values are shown at the top of the visualization.
              Remember that lower fitness values are better (we're minimizing skill standard deviation).
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Progress Bar</h4>
            <p>
              The progress bar shows the current iteration relative to the maximum number of iterations.
            </p>
          </div>
          <p>
            <strong>What to look for:</strong> Notice how Hill Climbing makes rapid improvements
            early on but may plateau as it approaches a local optimum. If random restarts are enabled,
            you might see sudden jumps in the solution quality as the algorithm restarts from a new
            random solution.
          </p>
        </div>
      )
    },
    SimulatedAnnealing: {
      overview: (
        <div className="algorithm-explanation">
          <h3>Simulated Annealing Algorithm</h3>
          <p>
            Simulated Annealing is a probabilistic optimization algorithm inspired by the annealing
            process in metallurgy, where metals are heated and then slowly cooled to reduce defects.
          </p>
          <p>
            In the Fantasy League context, Simulated Annealing works by:
          </p>
          <ol>
            <li>Starting with a random team assignment</li>
            <li>Exploring neighboring solutions by swapping players between teams</li>
            <li>Always accepting moves that improve the fitness</li>
            <li>Sometimes accepting worse moves with a probability that decreases over time</li>
            <li>Gradually reducing the "temperature" to decrease the acceptance of worse moves</li>
          </ol>
          <p>
            <strong>Strengths:</strong> Can escape local optima by accepting worse moves early in the search,
            often finding better solutions than Hill Climbing.
          </p>
          <p>
            <strong>Weaknesses:</strong> Requires careful tuning of the cooling schedule, and the
            probabilistic nature means results can vary between runs.
          </p>
        </div>
      ),
      operators: (
        <div className="algorithm-explanation">
          <h3>Simulated Annealing Operators</h3>
          <p>
            Simulated Annealing uses the following operators to navigate the search space:
          </p>
          <div className="operator-card">
            <h4>Swap Mutation</h4>
            <p>
              Similar to Hill Climbing, Simulated Annealing uses swap mutation to create
              neighboring solutions.
            </p>
            <p>
              <strong>How it works:</strong> Two teams are randomly selected, and one player from
              each team is swapped. The new solution is evaluated, and accepted based on the
              acceptance probability.
            </p>
          </div>
          <div className="operator-card">
            <h4>Acceptance Probability</h4>
            <p>
              The key difference from Hill Climbing is the acceptance probability function,
              which allows accepting worse solutions with a probability that decreases over time.
            </p>
            <p>
              <strong>How it works:</strong> The probability of accepting a worse solution is
              calculated as e^(-ΔE/T), where ΔE is the change in fitness and T is the current
              temperature. As temperature decreases, the probability of accepting worse moves
              also decreases.
            </p>
            <div className="operator-visualization">
              <div className="temperature-curve">
                <div className="curve-label">Temperature</div>
                <div className="curve-line"></div>
                <div className="curve-x-label">Iterations</div>
              </div>
              <div className="acceptance-curve">
                <div className="curve-label">Acceptance Probability</div>
                <div className="curve-line"></div>
                <div className="curve-x-label">Iterations</div>
              </div>
            </div>
          </div>
          <div className="operator-card">
            <h4>Cooling Schedule</h4>
            <p>
              The cooling schedule determines how the temperature decreases over time.
            </p>
            <p>
              <strong>How it works:</strong> The temperature is typically reduced by multiplying
              by a cooling rate (e.g., T = T * 0.95) after a certain number of iterations.
            </p>
          </div>
        </div>
      ),
      parameters: (
        <div className="algorithm-explanation">
          <h3>Simulated Annealing Parameters</h3>
          <p>
            The following parameters control the behavior of the Simulated Annealing algorithm:
          </p>
          <div className="parameter-card">
            <h4>Initial Temperature</h4>
            <p>
              The starting temperature of the annealing process.
            </p>
            <p>
              <strong>Impact:</strong> Higher values increase the initial acceptance probability
              of worse moves, allowing more exploration. Lower values make the algorithm more
              similar to Hill Climbing.
            </p>
            <p>
              <strong>Recommended range:</strong> 10-1000 depending on the fitness scale.
            </p>
          </div>
          <div className="parameter-card">
            <h4>Cooling Rate</h4>
            <p>
              The rate at which the temperature decreases.
            </p>
            <p>
              <strong>Impact:</strong> Higher values (closer to 1) result in slower cooling,
              allowing more exploration but increasing computational cost. Lower values result
              in faster cooling, potentially missing better solutions.
            </p>
            <p>
              <strong>Recommended range:</strong> 0.8-0.99
            </p>
          </div>
          <div className="parameter-card">
            <h4>Minimum Temperature</h4>
            <p>
              The temperature at which the algorithm stops.
            </p>
            <p>
              <strong>Impact:</strong> Lower values allow the algorithm to run longer,
              potentially finding better solutions but increasing computational cost.
            </p>
            <p>
              <strong>Recommended range:</strong> 0.001-1.0
            </p>
          </div>
          <div className="parameter-card">
            <h4>Max Iterations</h4>
            <p>
              The maximum number of iterations the algorithm will perform before stopping.
            </p>
            <p>
              <strong>Impact:</strong> Higher values allow more time for exploration but increase
              computational cost.
            </p>
            <p>
              <strong>Recommended range:</strong> 100-10,000 depending on problem complexity.
            </p>
          </div>
        </div>
      ),
      metrics: (
        <div className="algorithm-explanation">
          <h3>Simulated Annealing Metrics</h3>
          <p>
            The following metrics help evaluate the performance of Simulated Annealing:
          </p>
          <div className="metric-card">
            <h4>Fitness Value</h4>
            <p>
              The objective function value (skill standard deviation) that we're trying to minimize.
            </p>
            <p>
              <strong>Interpretation:</strong> Lower values indicate more balanced teams.
            </p>
          </div>
          <div className="metric-card">
            <h4>Temperature</h4>
            <p>
              The current temperature of the annealing process.
            </p>
            <p>
              <strong>Interpretation:</strong> Higher temperatures allow more exploration,
              while lower temperatures focus on exploitation.
            </p>
          </div>
          <div className="metric-card">
            <h4>Acceptance Rate</h4>
            <p>
              The percentage of proposed moves that are accepted.
            </p>
            <p>
              <strong>Interpretation:</strong> Should start high and gradually decrease as
              the temperature decreases.
            </p>
          </div>
          <div className="metric-card">
            <h4>Uphill Moves</h4>
            <p>
              The number of worse moves that are accepted.
            </p>
            <p>
              <strong>Interpretation:</strong> Should decrease as the temperature decreases.
              Too many uphill moves might indicate the initial temperature is too high.
            </p>
          </div>
          <div className="metric-card">
            <h4>Cooling Schedule Efficiency</h4>
            <p>
              How well the cooling schedule balances exploration and exploitation.
            </p>
            <p>
              <strong>Interpretation:</strong> Efficient cooling schedules show a smooth
              transition from exploration to exploitation.
            </p>
          </div>
        </div>
      ),
      visualization: (
        <div className="algorithm-explanation">
          <h3>Simulated Annealing Visualization Guide</h3>
          <p>
            The visualization shows how Simulated Annealing optimizes team assignments:
          </p>
          <div className="visualization-guide">
            <h4>Team Flags</h4>
            <p>
              Each colored flag represents a team. Players are shown as cards beneath their team flag.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Player Movements</h4>
            <p>
              When the algorithm swaps players between teams, you'll see the player cards move
              from one team to another. Unlike Hill Climbing, Simulated Annealing sometimes
              accepts moves that make the solution worse, especially early in the search.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Temperature Display</h4>
            <p>
              The current temperature is shown, decreasing over time according to the cooling schedule.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Acceptance Probability</h4>
            <p>
              For worse moves, the acceptance probability is shown, decreasing as the temperature decreases.
            </p>
          </div>
          <p>
            <strong>What to look for:</strong> Notice how Simulated Annealing accepts some worse moves
            early in the search (when the temperature is high), allowing it to escape local optima.
            As the temperature decreases, the algorithm becomes more selective, eventually behaving
            like Hill Climbing.
          </p>
        </div>
      )
    },
    GeneticAlgorithm: {
      overview: (
        <div className="algorithm-explanation">
          <h3>Genetic Algorithm</h3>
          <p>
            Genetic Algorithms are population-based optimization algorithms inspired by natural selection
            and genetics. They evolve a population of solutions over generations, using selection,
            crossover, and mutation operators.
          </p>
          <p>
            In the Fantasy League context, Genetic Algorithms work by:
          </p>
          <ol>
            <li>Creating an initial population of random team assignments</li>
            <li>Evaluating the fitness of each solution</li>
            <li>Selecting parent solutions based on their fitness</li>
            <li>Creating offspring through crossover (combining parts of parents)</li>
            <li>Applying mutation to introduce diversity</li>
            <li>Replacing the old population with the new generation</li>
            <li>Repeating until a termination condition is met</li>
          </ol>
          <p>
            <strong>Strengths:</strong> Can explore multiple regions of the search space simultaneously,
            often finding better solutions than single-solution methods like Hill Climbing and Simulated Annealing.
          </p>
          <p>
            <strong>Weaknesses:</strong> Requires more parameters to tune, and the population-based approach
            increases computational cost.
          </p>
        </div>
      ),
      operators: (
        <div className="algorithm-explanation">
          <h3>Genetic Algorithm Operators</h3>
          <p>
            Genetic Algorithms use the following operators to navigate the search space:
          </p>
          <div className="operator-card">
            <h4>Selection</h4>
            <p>
              Selection operators choose parent solutions for reproduction based on their fitness.
            </p>
            <p>
              <strong>Tournament Selection:</strong> Randomly selects a subset of solutions and
              chooses the best one as a parent.
            </p>
            <p>
              <strong>Roulette Wheel Selection:</strong> Selects parents with probability proportional
              to their fitness.
            </p>
            <p>
              <strong>Ranking Selection:</strong> Selects parents based on their rank in the population,
              rather than their absolute fitness.
            </p>
          </div>
          <div className="operator-card">
            <h4>Crossover</h4>
            <p>
              Crossover operators combine parts of two parent solutions to create offspring.
            </p>
            <p>
              <strong>One-Point Crossover:</strong> Selects a random point and swaps all players
              after that point between the two parents.
            </p>
            <div className="operator-visualization">
              <div className="team-before">
                <div className="team-label">Parent 1</div>
                <div className="player-list">
                  <div className="player">Player A</div>
                  <div className="player">Player B</div>
                  <div className="player highlighted">Player C</div>
                  <div className="player highlighted">Player D</div>
                </div>
              </div>
              <div className="swap-arrow">↓</div>
              <div className="team-before">
                <div className="team-label">Parent 2</div>
                <div className="player-list">
                  <div className="player">Player E</div>
                  <div className="player">Player F</div>
                  <div className="player highlighted">Player G</div>
                  <div className="player highlighted">Player H</div>
                </div>
              </div>
              <div className="swap-arrow">↓</div>
              <div className="team-after">
                <div className="team-label">Offspring 1</div>
                <div className="player-list">
                  <div className="player">Player A</div>
                  <div className="player">Player B</div>
                  <div className="player highlighted">Player G</div>
                  <div className="player highlighted">Player H</div>
                </div>
              </div>
              <div className="team-after">
                <div className="team-label">Offspring 2</div>
                <div className="player-list">
                  <div className="player">Player E</div>
                  <div className="player">Player F</div>
                  <div className="player highlighted">Player C</div>
                  <div className="player highlighted">Player D</div>
                </div>
              </div>
            </div>
            <p>
              <strong>Two-Point Crossover:</strong> Selects two random points and swaps the players
              between those points.
            </p>
            <p>
              <strong>Uniform Crossover:</strong> For each player position, randomly decides which
              parent to inherit from.
            </p>
          </div>
          <div className="operator-card">
            <h4>Mutation</h4>
            <p>
              Mutation operators introduce small random changes to solutions to maintain diversity.
            </p>
            <p>
              <strong>Swap Mutation:</strong> Randomly selects two players and swaps them.
            </p>
          </div>
        </div>
      ),
      parameters: (
        <div className="algorithm-explanation">
          <h3>Genetic Algorithm Parameters</h3>
          <p>
            The following parameters control the behavior of the Genetic Algorithm:
          </p>
          <div className="parameter-card">
            <h4>Population Size</h4>
            <p>
              The number of solutions in the population.
            </p>
            <p>
              <strong>Impact:</strong> Larger populations provide more diversity but increase
              computational cost. Smaller populations may converge prematurely.
            </p>
            <p>
              <strong>Recommended range:</strong> 20-200 depending on problem complexity.
            </p>
          </div>
          <div className="parameter-card">
            <h4>Max Generations</h4>
            <p>
              The maximum number of generations the algorithm will run.
            </p>
            <p>
              <strong>Impact:</strong> More generations allow more time for evolution but increase
              computational cost.
            </p>
            <p>
              <strong>Recommended range:</strong> 50-1000 depending on problem complexity.
            </p>
          </div>
          <div className="parameter-card">
            <h4>Crossover Rate</h4>
            <p>
              The probability of applying crossover to selected parents.
            </p>
            <p>
              <strong>Impact:</strong> Higher values increase the rate of recombination,
              potentially accelerating convergence. Lower values preserve more of the original
              solutions.
            </p>
            <p>
              <strong>Recommended range:</strong> 0.6-0.9
            </p>
          </div>
          <div className="parameter-card">
            <h4>Mutation Rate</h4>
            <p>
              The probability of applying mutation to each offspring.
            </p>
            <p>
              <strong>Impact:</strong> Higher values increase diversity but may disrupt good
              solutions. Lower values may lead to premature convergence.
            </p>
            <p>
              <strong>Recommended range:</strong> 0.01-0.2
            </p>
          </div>
          <div className="parameter-card">
            <h4>Selection Method</h4>
            <p>
              The method used to select parents for reproduction.
            </p>
            <p>
              <strong>Impact:</strong> Different selection methods create different selection
              pressures, affecting the balance between exploration and exploitation.
            </p>
            <p>
              <strong>Options:</strong> Tournament, Roulette Wheel, Ranking
            </p>
          </div>
          <div className="parameter-card">
            <h4>Crossover Method</h4>
            <p>
              The method used to combine parents to create offspring.
            </p>
            <p>
              <strong>Impact:</strong> Different crossover methods create different patterns
              of inheritance, affecting how solutions are combined.
            </p>
            <p>
              <strong>Options:</strong> One-Point, Two-Point, Uniform
            </p>
          </div>
        </div>
      ),
      metrics: (
        <div className="algorithm-explanation">
          <h3>Genetic Algorithm Metrics</h3>
          <p>
            The following metrics help evaluate the performance of Genetic Algorithms:
          </p>
          <div className="metric-card">
            <h4>Best Fitness</h4>
            <p>
              The fitness of the best solution in the current population.
            </p>
            <p>
              <strong>Interpretation:</strong> Should improve over generations, with occasional
              plateaus as the algorithm explores.
            </p>
          </div>
          <div className="metric-card">
            <h4>Average Fitness</h4>
            <p>
              The average fitness of all solutions in the population.
            </p>
            <p>
              <strong>Interpretation:</strong> Should improve over generations, but more slowly
              than the best fitness. The gap between best and average fitness indicates the
              diversity of the population.
            </p>
          </div>
          <div className="metric-card">
            <h4>Population Diversity</h4>
            <p>
              A measure of how different the solutions in the population are from each other.
            </p>
            <p>
              <strong>Interpretation:</strong> Should start high and gradually decrease as the
              population converges. Too rapid a decrease may indicate premature convergence.
            </p>
          </div>
          <div className="metric-card">
            <h4>Selection Pressure</h4>
            <p>
              How strongly the selection favors better solutions.
            </p>
            <p>
              <strong>Interpretation:</strong> Higher values indicate more exploitation,
              while lower values indicate more exploration.
            </p>
          </div>
          <div className="metric-card">
            <h4>Crossover Success Rate</h4>
            <p>
              The percentage of crossovers that produce offspring better than their parents.
            </p>
            <p>
              <strong>Interpretation:</strong> Higher values indicate effective recombination.
            </p>
          </div>
          <div className="metric-card">
            <h4>Mutation Impact</h4>
            <p>
              The effect of mutation on solution quality.
            </p>
            <p>
              <strong>Interpretation:</strong> Positive values indicate beneficial mutations,
              while negative values indicate harmful mutations.
            </p>
          </div>
        </div>
      ),
      visualization: (
        <div className="algorithm-explanation">
          <h3>Genetic Algorithm Visualization Guide</h3>
          <p>
            The visualization shows how Genetic Algorithms optimize team assignments:
          </p>
          <div className="visualization-guide">
            <h4>Population</h4>
            <p>
              The visualization shows a subset of the population, with each solution represented
              as a set of team flags and player cards.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Selection</h4>
            <p>
              When parents are selected for reproduction, their team flags are highlighted.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Crossover</h4>
            <p>
              When crossover occurs, you'll see player cards moving between parent solutions
              to create offspring.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Mutation</h4>
            <p>
              When mutation occurs, you'll see player cards swapping positions within a solution.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Generation Counter</h4>
            <p>
              The current generation is displayed, along with the maximum number of generations.
            </p>
          </div>
          <p>
            <strong>What to look for:</strong> Notice how the population evolves over generations,
            with the best solutions surviving and reproducing. The diversity of the population
            should gradually decrease as the algorithm converges on good solutions.
          </p>
        </div>
      )
    },
    IslandGA: {
      overview: (
        <div className="algorithm-explanation">
          <h3>Island Genetic Algorithm</h3>
          <p>
            Island Genetic Algorithms are an extension of traditional Genetic Algorithms that divide
            the population into multiple isolated "islands" that evolve independently, with occasional
            migration of individuals between islands.
          </p>
          <p>
            In the Fantasy League context, Island Genetic Algorithms work by:
          </p>
          <ol>
            <li>Dividing the population into multiple islands</li>
            <li>Evolving each island independently using standard GA operations</li>
            <li>Periodically migrating individuals between islands according to a migration topology</li>
            <li>Continuing until a termination condition is met</li>
          </ol>
          <p>
            <strong>Strengths:</strong> Maintains higher population diversity by isolating subpopulations,
            often finding better solutions than standard GAs. The parallel nature also makes them suitable
            for distributed computing.
          </p>
          <p>
            <strong>Weaknesses:</strong> Adds complexity with additional parameters to tune (number of islands,
            migration frequency, migration topology, etc.).
          </p>
        </div>
      ),
      operators: (
        <div className="algorithm-explanation">
          <h3>Island Genetic Algorithm Operators</h3>
          <p>
            Island Genetic Algorithms use all the standard GA operators (selection, crossover, mutation),
            plus the following island-specific operators:
          </p>
          <div className="operator-card">
            <h4>Migration</h4>
            <p>
              Migration is the process of moving individuals between islands.
            </p>
            <p>
              <strong>How it works:</strong> At regular intervals (the migration frequency),
              a subset of individuals (determined by the migration rate) is selected from each
              island and sent to other islands according to the migration topology.
            </p>
          </div>
          <div className="operator-card">
            <h4>Migration Topologies</h4>
            <p>
              The migration topology determines which islands can exchange individuals.
            </p>
            <p>
              <strong>Ring Topology:</strong> Islands are arranged in a ring, and migration
              occurs only between adjacent islands.
            </p>
            <div className="operator-visualization">
              <div className="island-topology ring-topology">
                <div className="island">Island 1</div>
                <div className="island">Island 2</div>
                <div className="island">Island 3</div>
                <div className="island">Island 4</div>
                <div className="migration-arrow">→</div>
                <div className="migration-arrow">→</div>
                <div className="migration-arrow">→</div>
                <div className="migration-arrow">→</div>
              </div>
            </div>
            <p>
              <strong>Star Topology:</strong> One central island exchanges individuals with
              all other islands.
            </p>
            <div className="operator-visualization">
              <div className="island-topology star-topology">
                <div className="island central">Island 1</div>
                <div className="island">Island 2</div>
                <div className="island">Island 3</div>
                <div className="island">Island 4</div>
                <div className="migration-arrow">↔</div>
                <div className="migration-arrow">↔</div>
                <div className="migration-arrow">↔</div>
              </div>
            </div>
            <p>
              <strong>Fully Connected Topology:</strong> Every island can exchange individuals
              with every other island.
            </p>
            <div className="operator-visualization">
              <div className="island-topology fully-connected-topology">
                <div className="island">Island 1</div>
                <div className="island">Island 2</div>
                <div className="island">Island 3</div>
                <div className="island">Island 4</div>
                <div className="migration-arrow">↔</div>
                <div className="migration-arrow">↔</div>
                <div className="migration-arrow">↔</div>
                <div className="migration-arrow">↔</div>
                <div className="migration-arrow">↔</div>
                <div className="migration-arrow">↔</div>
              </div>
            </div>
          </div>
          <div className="operator-card">
            <h4>Migration Policies</h4>
            <p>
              Migration policies determine which individuals are selected for migration and
              how they are integrated into the destination island.
            </p>
            <p>
              <strong>Best Individuals:</strong> The best individuals from each island are
              selected for migration.
            </p>
            <p>
              <strong>Random Individuals:</strong> Random individuals are selected for migration.
            </p>
            <p>
              <strong>Replace Worst:</strong> Migrants replace the worst individuals in the
              destination island.
            </p>
            <p>
              <strong>Replace Random:</strong> Migrants replace random individuals in the
              destination island.
            </p>
          </div>
        </div>
      ),
      parameters: (
        <div className="algorithm-explanation">
          <h3>Island Genetic Algorithm Parameters</h3>
          <p>
            In addition to the standard GA parameters, Island GAs have the following parameters:
          </p>
          <div className="parameter-card">
            <h4>Number of Islands</h4>
            <p>
              The number of separate subpopulations.
            </p>
            <p>
              <strong>Impact:</strong> More islands increase diversity but divide the total
              population size. Too many islands with small populations may limit the effectiveness
              of each island's evolution.
            </p>
            <p>
              <strong>Recommended range:</strong> 3-10 depending on total population size.
            </p>
          </div>
          <div className="parameter-card">
            <h4>Island Population Size</h4>
            <p>
              The number of individuals in each island.
            </p>
            <p>
              <strong>Impact:</strong> Larger island populations provide more diversity within
              each island but may reduce the benefit of isolation.
            </p>
            <p>
              <strong>Recommended range:</strong> 10-50 depending on problem complexity.
            </p>
          </div>
          <div className="parameter-card">
            <h4>Migration Frequency</h4>
            <p>
              How often migration occurs, measured in generations.
            </p>
            <p>
              <strong>Impact:</strong> Higher values allow islands to evolve more independently,
              potentially developing unique solutions. Lower values increase the exchange of
              genetic material between islands.
            </p>
            <p>
              <strong>Recommended range:</strong> 5-20 generations.
            </p>
          </div>
          <div className="parameter-card">
            <h4>Migration Rate</h4>
            <p>
              The percentage of each island's population that migrates.
            </p>
            <p>
              <strong>Impact:</strong> Higher values increase the exchange of genetic material
              but may disrupt the unique evolution of each island. Lower values maintain more
              isolation.
            </p>
            <p>
              <strong>Recommended range:</strong> 0.05-0.3 (5-30%)
            </p>
          </div>
          <div className="parameter-card">
            <h4>Migration Topology</h4>
            <p>
              The pattern of connections between islands.
            </p>
            <p>
              <strong>Impact:</strong> Different topologies create different patterns of
              genetic exchange, affecting how solutions spread through the population.
            </p>
            <p>
              <strong>Options:</strong> Ring, Star, Fully Connected
            </p>
          </div>
        </div>
      ),
      metrics: (
        <div className="algorithm-explanation">
          <h3>Island Genetic Algorithm Metrics</h3>
          <p>
            In addition to the standard GA metrics, Island GAs have the following metrics:
          </p>
          <div className="metric-card">
            <h4>Inter-Island Diversity</h4>
            <p>
              A measure of how different the solutions are between islands.
            </p>
            <p>
              <strong>Interpretation:</strong> Higher values indicate that islands are evolving
              in different directions, potentially exploring different regions of the search space.
            </p>
          </div>
          <div className="metric-card">
            <h4>Intra-Island Diversity</h4>
            <p>
              A measure of how different the solutions are within each island.
            </p>
            <p>
              <strong>Interpretation:</strong> Should start high and gradually decrease as each
              island converges. Different islands may converge at different rates.
            </p>
          </div>
          <div className="metric-card">
            <h4>Migration Impact</h4>
            <p>
              The effect of migration on solution quality.
            </p>
            <p>
              <strong>Interpretation:</strong> Positive values indicate beneficial migrations,
              while negative values indicate harmful migrations.
            </p>
          </div>
          <div className="metric-card">
            <h4>Island Convergence Rates</h4>
            <p>
              How quickly each island converges to a solution.
            </p>
            <p>
              <strong>Interpretation:</strong> Different convergence rates indicate that islands
              are exploring different regions of the search space.
            </p>
          </div>
          <div className="metric-card">
            <h4>Best Island</h4>
            <p>
              Which island consistently produces the best solutions.
            </p>
            <p>
              <strong>Interpretation:</strong> If one island consistently outperforms others,
              it may have found a particularly promising region of the search space.
            </p>
          </div>
        </div>
      ),
      visualization: (
        <div className="algorithm-explanation">
          <h3>Island Genetic Algorithm Visualization Guide</h3>
          <p>
            The visualization shows how Island Genetic Algorithms optimize team assignments:
          </p>
          <div className="visualization-guide">
            <h4>Islands</h4>
            <p>
              Each island is represented as a separate group of team flags and player cards.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Migration</h4>
            <p>
              When migration occurs, you'll see player cards moving between islands according
              to the migration topology.
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Island Evolution</h4>
            <p>
              Each island evolves independently using standard GA operations (selection,
              crossover, mutation).
            </p>
          </div>
          <div className="visualization-guide">
            <h4>Migration Frequency</h4>
            <p>
              The migration frequency is displayed, showing how many generations occur
              between migrations.
            </p>
          </div>
          <p>
            <strong>What to look for:</strong> Notice how different islands may evolve in
            different directions, potentially finding different local optima. When migration
            occurs, good solutions can spread from one island to another, potentially leading
            to better overall solutions.
          </p>
        </div>
      )
    }
  };
  
  // Get content for the selected algorithm and tab
  const getContent = () => {
    if (!algorithmContent[selectedAlgorithm]) {
      return (
        <div className="algorithm-explanation">
          <h3>Algorithm Not Selected</h3>
          <p>Please select an algorithm to view its explanation.</p>
        </div>
      );
    }
    
    return algorithmContent[selectedAlgorithm][activeTab] || (
      <div className="algorithm-explanation">
        <h3>Content Not Available</h3>
        <p>Explanation for this tab is not available.</p>
      </div>
    );
  };
  
  return (
    <div className="algorithm-explanation-tabs">
      <div className="tabs-header">
        {Object.entries(tabs).map(([tabId, tabName]) => (
          <button
            key={tabId}
            className={`tab-button ${activeTab === tabId ? 'active' : ''}`}
            onClick={() => setActiveTab(tabId)}
          >
            {tabName}
          </button>
        ))}
      </div>
      <div className="tab-content">
        {getContent()}
      </div>
    </div>
  );
};

export default AlgorithmExplanationTabs;
