/**
 * HillClimbing visualizer component for the educational website.
 *
 * This component provides an interactive visualization of how the Hill Climbing
 * algorithm works on the Fantasy League problem.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, Button, Form, Row, Col, Badge, ProgressBar, Table } from 'react-bootstrap';

const HillClimbingVisualizer = () => {
  // State for visualization
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentStep, setCurrentStep] = useState(0);
  const [showExplanation, setShowExplanation] = useState(true);
  
  // State for algorithm parameters
  const [maxIterations, setMaxIterations] = useState(100);
  const [maxNoImprovement, setMaxNoImprovement] = useState(20);
  
  // State for algorithm execution
  const [steps, setSteps] = useState([]);
  const [currentSolution, setCurrentSolution] = useState(null);
  const [bestSolution, setBestSolution] = useState(null);
  const [neighborhoodSize, setNeighborhoodSize] = useState(10);
  
  // Animation reference
  const animationRef = useRef(null);
  
  // Initialize with dummy data
  useEffect(() => {
    generateDummyData();
  }, []);
  
  // Generate dummy data for visualization
  const generateDummyData = () => {
    // Create initial solution
    const initialSolution = {
      players: [
        { id: 1, name: 'Player 1', position: 'GK', skill: 85, cost: 10 },
        { id: 2, name: 'Player 2', position: 'DEF', skill: 82, cost: 8 },
        { id: 3, name: 'Player 3', position: 'DEF', skill: 79, cost: 7 },
        { id: 4, name: 'Player 4', position: 'DEF', skill: 81, cost: 9 },
        { id: 5, name: 'Player 5', position: 'DEF', skill: 78, cost: 6 },
        { id: 6, name: 'Player 6', position: 'MID', skill: 84, cost: 12 },
        { id: 7, name: 'Player 7', position: 'MID', skill: 83, cost: 11 },
        { id: 8, name: 'Player 8', position: 'MID', skill: 80, cost: 9 },
        { id: 9, name: 'Player 9', position: 'MID', skill: 77, cost: 7 },
        { id: 10, name: 'Player 10', position: 'FWD', skill: 86, cost: 14 },
        { id: 11, name: 'Player 11', position: 'FWD', skill: 82, cost: 10 }
      ],
      fitness: 10.5,
      cost: 103,
      stdDev: 2.94
    };
    
    setCurrentSolution(initialSolution);
    setBestSolution(initialSolution);
    
    // Generate steps
    const generatedSteps = [];
    let currentFitness = initialSolution.fitness;
    let iterationsWithoutImprovement = 0;
    
    for (let i = 0; i < maxIterations; i++) {
      // Generate neighbors
      const neighbors = Array.from({ length: neighborhoodSize }, (_, j) => {
        const neighborFitness = currentFitness - (Math.random() * 0.5) * Math.exp(-i / 20);
        return {
          id: `neighbor-${i}-${j}`,
          fitness: neighborFitness,
          isBest: false
        };
      });
      
      // Sort neighbors by fitness (lower is better)
      neighbors.sort((a, b) => a.fitness - b.fitness);
      
      // Mark best neighbor
      neighbors[0].isBest = true;
      
      // Determine if improvement
      const bestNeighborFitness = neighbors[0].fitness;
      const improvement = bestNeighborFitness < currentFitness;
      
      if (improvement) {
        iterationsWithoutImprovement = 0;
      } else {
        iterationsWithoutImprovement++;
      }
      
      // Create step
      const step = {
        iteration: i,
        currentFitness,
        neighbors,
        bestNeighborFitness,
        improvement,
        iterationsWithoutImprovement,
        earlyStop: iterationsWithoutImprovement >= maxNoImprovement
      };
      
      generatedSteps.push(step);
      
      // Update current fitness if improved
      if (improvement) {
        currentFitness = bestNeighborFitness;
      }
      
      // Check early stopping
      if (iterationsWithoutImprovement >= maxNoImprovement) {
        break;
      }
    }
    
    setSteps(generatedSteps);
  };
  
  // Start animation
  const startAnimation = () => {
    setIsRunning(true);
    setIsPaused(false);
    
    if (currentStep >= steps.length) {
      setCurrentStep(0);
    }
    
    animateStep();
  };
  
  // Animate a single step
  const animateStep = () => {
    if (currentStep < steps.length) {
      // Update current solution based on step
      const step = steps[currentStep];
      
      if (step.improvement) {
        const newSolution = { ...currentSolution, fitness: step.bestNeighborFitness };
        setCurrentSolution(newSolution);
        
        if (newSolution.fitness < bestSolution.fitness) {
          setBestSolution(newSolution);
        }
      }
      
      // Schedule next step
      const delay = 2000 / speed;
      animationRef.current = setTimeout(() => {
        setCurrentStep(prevStep => prevStep + 1);
        
        if (currentStep + 1 < steps.length && !isPaused) {
          animateStep();
        } else {
          setIsRunning(false);
        }
      }, delay);
    } else {
      setIsRunning(false);
    }
  };
  
  // Pause animation
  const pauseAnimation = () => {
    setIsPaused(true);
    setIsRunning(false);
    
    if (animationRef.current) {
      clearTimeout(animationRef.current);
    }
  };
  
  // Step forward
  const stepForward = () => {
    if (currentStep < steps.length) {
      // Update current solution based on step
      const step = steps[currentStep];
      
      if (step.improvement) {
        const newSolution = { ...currentSolution, fitness: step.bestNeighborFitness };
        setCurrentSolution(newSolution);
        
        if (newSolution.fitness < bestSolution.fitness) {
          setBestSolution(newSolution);
        }
      }
      
      setCurrentStep(prevStep => prevStep + 1);
    }
  };
  
  // Reset animation
  const resetAnimation = () => {
    setIsRunning(false);
    setIsPaused(false);
    setCurrentStep(0);
    
    if (animationRef.current) {
      clearTimeout(animationRef.current);
    }
    
    // Reset solutions
    if (steps.length > 0) {
      const initialSolution = { ...currentSolution, fitness: steps[0].currentFitness };
      setCurrentSolution(initialSolution);
      setBestSolution(initialSolution);
    }
  };
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, []);
  
  // Regenerate data when parameters change
  useEffect(() => {
    resetAnimation();
    generateDummyData();
  }, [maxIterations, maxNoImprovement, neighborhoodSize]);
  
  return (
    <div className="hill-climbing-visualizer">
      <Card className="mb-4">
        <Card.Header>
          <h3>Hill Climbing Algorithm Visualization</h3>
        </Card.Header>
        <Card.Body>
          <Row className="mb-4">
            <Col md={6}>
              <h4>Algorithm Parameters</h4>
              <Form>
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Max Iterations:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={maxIterations} 
                      onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                      min={10}
                      max={1000}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Max No Improvement:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={maxNoImprovement} 
                      onChange={(e) => setMaxNoImprovement(parseInt(e.target.value))}
                      min={5}
                      max={100}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Neighborhood Size:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={neighborhoodSize} 
                      onChange={(e) => setNeighborhoodSize(parseInt(e.target.value))}
                      min={5}
                      max={50}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
              </Form>
            </Col>
            
            <Col md={6}>
              <h4>Visualization Controls</h4>
              <div className="d-flex mb-3">
                <Button 
                  variant="success" 
                  onClick={startAnimation} 
                  disabled={isRunning || currentStep >= steps.length}
                  className="me-2"
                >
                  {isPaused ? 'Resume' : 'Start'}
                </Button>
                
                <Button 
                  variant="warning" 
                  onClick={pauseAnimation} 
                  disabled={!isRunning}
                  className="me-2"
                >
                  Pause
                </Button>
                
                <Button 
                  variant="info" 
                  onClick={stepForward} 
                  disabled={isRunning || currentStep >= steps.length}
                  className="me-2"
                >
                  Step
                </Button>
                
                <Button 
                  variant="danger" 
                  onClick={resetAnimation}
                  disabled={isRunning && !isPaused}
                >
                  Reset
                </Button>
              </div>
              
              <Form.Group className="mb-3">
                <Form.Label>Animation Speed:</Form.Label>
                <Form.Range 
                  min={0.5}
                  max={5}
                  step={0.5}
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                />
                <div className="d-flex justify-content-between">
                  <small>Slow</small>
                  <small>Fast</small>
                </div>
              </Form.Group>
            </Col>
          </Row>
          
          <Row className="mb-4">
            <Col>
              <h4>Algorithm Progress</h4>
              <ProgressBar 
                now={(currentStep / (steps.length || 1)) * 100} 
                label={`${currentStep} / ${steps.length}`}
                variant="primary"
                className="mb-3"
              />
              
              <div className="d-flex justify-content-between mb-3">
                <div>
                  <strong>Current Iteration:</strong> {currentStep < steps.length ? steps[currentStep].iteration : 'N/A'}
                </div>
                <div>
                  <strong>Iterations without Improvement:</strong> {
                    currentStep < steps.length ? 
                    steps[currentStep].iterationsWithoutImprovement : 'N/A'
                  }
                </div>
                <div>
                  <strong>Status:</strong> {
                    currentStep >= steps.length ? 
                    <Badge bg="success">Completed</Badge> : 
                    isRunning ? 
                    <Badge bg="primary">Running</Badge> : 
                    <Badge bg="secondary">Idle</Badge>
                  }
                </div>
              </div>
            </Col>
          </Row>
          
          <Row className="mb-4">
            <Col md={6}>
              <Card>
                <Card.Header>
                  <h5>Current Solution</h5>
                </Card.Header>
                <Card.Body>
                  <div className="mb-3">
                    <strong>Fitness:</strong> {currentSolution?.fitness.toFixed(4)}
                  </div>
                  <div className="mb-3">
                    <strong>Cost:</strong> {currentSolution?.cost}
                  </div>
                  <div className="mb-3">
                    <strong>Standard Deviation:</strong> {currentSolution?.stdDev.toFixed(4)}
                  </div>
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card>
                <Card.Header>
                  <h5>Best Solution</h5>
                </Card.Header>
                <Card.Body>
                  <div className="mb-3">
                    <strong>Fitness:</strong> {bestSolution?.fitness.toFixed(4)}
                  </div>
                  <div className="mb-3">
                    <strong>Cost:</strong> {bestSolution?.cost}
                  </div>
                  <div className="mb-3">
                    <strong>Standard Deviation:</strong> {bestSolution?.stdDev.toFixed(4)}
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          {currentStep < steps.length && (
            <Row className="mb-4">
              <Col>
                <h4>Current Neighborhood</h4>
                <Table striped bordered hover size="sm">
                  <thead>
                    <tr>
                      <th>Neighbor</th>
                      <th>Fitness</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {steps[currentStep].neighbors.map((neighbor, index) => (
                      <tr key={neighbor.id} className={neighbor.isBest ? 'table-success' : ''}>
                        <td>Neighbor {index + 1}</td>
                        <td>{neighbor.fitness.toFixed(4)}</td>
                        <td>
                          {neighbor.isBest ? 
                            <Badge bg="success">Best Neighbor</Badge> : 
                            <Badge bg="secondary">Regular</Badge>
                          }
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </Col>
            </Row>
          )}
          
          <Row>
            <Col>
              <Form.Check 
                type="switch"
                id="explanation-switch"
                label="Show Algorithm Explanation"
                checked={showExplanation}
                onChange={(e) => setShowExplanation(e.target.checked)}
                className="mb-3"
              />
              
              {showExplanation && (
                <Card bg="light">
                  <Card.Body>
                    <h5>How Hill Climbing Works</h5>
                    <p>
                      Hill Climbing is a local search algorithm that starts with an initial solution and iteratively moves to better neighboring solutions.
                    </p>
                    <ol>
                      <li>Start with a random initial solution</li>
                      <li>Evaluate all neighboring solutions</li>
                      <li>Move to the best neighbor if it's better than the current solution</li>
                      <li>Repeat until no better neighbor is found or a stopping criterion is met</li>
                    </ol>
                    <h5>Key Metrics</h5>
                    <ul>
                      <li><strong>Neighbors Generated:</strong> Total number of neighboring solutions created</li>
                      <li><strong>Neighbors Evaluated:</strong> Total number of neighboring solutions evaluated</li>
                      <li><strong>Local Optima Count:</strong> Number of local optima encountered</li>
                      <li><strong>Plateau Length:</strong> Length of plateaus (regions with equal fitness) encountered</li>
                      <li><strong>Improvement Rate:</strong> Rate of improvement per neighbor evaluated</li>
                    </ul>
                    <h5>Strengths and Weaknesses</h5>
                    <p><strong>Strengths:</strong> Simple to implement, fast for small problems, good for fine-tuning solutions</p>
                    <p><strong>Weaknesses:</strong> Gets stuck in local optima, performance depends heavily on initial solution</p>
                  </Card.Body>
                </Card>
              )}
            </Col>
          </Row>
        </Card.Body>
      </Card>
    </div>
  );
};

export default HillClimbingVisualizer;
