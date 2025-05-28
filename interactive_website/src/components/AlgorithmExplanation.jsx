import React from 'react';
import PropTypes from 'prop-types';

const AlgorithmExplanation = ({ algorithm }) => {
  // Algorithm explanations
  const explanations = {
    HillClimbing: {
      title: 'Hill Climbing Algorithm',
      description: `
        Hill Climbing is a local search optimization algorithm that starts with an arbitrary solution and 
        iteratively makes small changes to improve it. It's like climbing a hill - always moving upward 
        until you reach a peak.
      `,
      steps: [
        'Start with a random initial solution',
        'Evaluate the current solution',
        'Generate neighboring solutions by making small changes',
        'If a neighbor is better, move to it and repeat',
        'If no neighbor is better, stop (local optimum reached)'
      ],
      strengths: [
        'Simple to implement and understand',
        'Works well for many problems with smooth fitness landscapes',
        'Requires minimal memory'
      ],
      weaknesses: [
        'Can get stuck in local optima',
        'Performance depends heavily on the initial solution',
        'Not suitable for complex multimodal problems'
      ]
    },
    SimulatedAnnealing: {
      title: 'Simulated Annealing Algorithm',
      description: `
        Simulated Annealing is inspired by the annealing process in metallurgy. It allows occasional "bad" 
        moves to escape local optima, with the probability of accepting worse solutions decreasing over time.
      `,
      steps: [
        'Start with a random initial solution and high temperature',
        'Generate a neighboring solution by making a small change',
        'If the neighbor is better, always move to it',
        'If the neighbor is worse, move to it with a probability based on temperature',
        'Decrease temperature according to cooling schedule',
        'Repeat until temperature reaches minimum or no improvement is found'
      ],
      strengths: [
        'Can escape local optima',
        'Proven to converge to global optimum given enough time',
        'Works well for complex problems with many local optima'
      ],
      weaknesses: [
        'Requires careful tuning of parameters',
        'Slower convergence than hill climbing for simple problems',
        'Performance depends on cooling schedule'
      ]
    },
    GeneticAlgorithm: {
      title: 'Genetic Algorithm',
      description: `
        Genetic Algorithms are inspired by natural selection. They maintain a population of solutions 
        and evolve them through selection, crossover, and mutation operations.
      `,
      steps: [
        'Initialize a population of random solutions',
        'Evaluate the fitness of each solution',
        'Select parents based on fitness (better solutions have higher chance)',
        'Create offspring through crossover (combining parts of parents)',
        'Apply mutation to introduce diversity',
        'Replace the old population with the new generation',
        'Repeat until stopping criteria are met'
      ],
      strengths: [
        'Can find good solutions to complex problems',
        'Maintains diversity through population-based approach',
        'Parallelizable and suitable for multi-objective optimization'
      ],
      weaknesses: [
        'Requires more computational resources than local search methods',
        'Many parameters to tune',
        'May converge prematurely to suboptimal solutions'
      ]
    },
    IslandGeneticAlgorithm: {
      title: 'Island Genetic Algorithm',
      description: `
        Island Genetic Algorithms extend the standard GA by dividing the population into multiple 
        "islands" that evolve independently, with occasional migration between islands.
      `,
      steps: [
        'Divide the population into multiple subpopulations (islands)',
        'Evolve each island independently using standard GA operations',
        'Periodically migrate individuals between islands',
        'Continue evolution until stopping criteria are met'
      ],
      strengths: [
        'Maintains higher diversity than standard GA',
        'Can explore different regions of the search space simultaneously',
        'Naturally parallelizable',
        'Less prone to premature convergence'
      ],
      weaknesses: [
        'More complex to implement and tune',
        'Requires additional parameters (migration rate, interval, topology)',
        'May require larger total population size'
      ]
    }
  };

  // Get explanation for selected algorithm
  const explanation = explanations[algorithm] || {
    title: 'Algorithm Explanation',
    description: 'Select an algorithm to see its explanation.',
    steps: [],
    strengths: [],
    weaknesses: []
  };

  return (
    <div className="algorithm-explanation">
      <h2>{explanation.title}</h2>
      
      <p>{explanation.description}</p>
      
      <h3>How it works:</h3>
      <ol>
        {explanation.steps.map((step, index) => (
          <li key={index}>{step}</li>
        ))}
      </ol>
      
      <h3>Strengths:</h3>
      <ul>
        {explanation.strengths.map((strength, index) => (
          <li key={index}>{strength}</li>
        ))}
      </ul>
      
      <h3>Limitations:</h3>
      <ul>
        {explanation.weaknesses.map((weakness, index) => (
          <li key={index}>{weakness}</li>
        ))}
      </ul>
      
      <h3>Application to Fantasy League Optimization:</h3>
      <p>
        In the Fantasy League Optimization problem, we're trying to assign players to teams 
        to maximize overall skill while respecting constraints like budget limits and position requirements.
        This algorithm searches through possible team assignments to find the best combination.
      </p>
    </div>
  );
};

AlgorithmExplanation.propTypes = {
  algorithm: PropTypes.string.isRequired
};

export default AlgorithmExplanation;
