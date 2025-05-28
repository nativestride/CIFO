/**
 * SimulatedAnnealing visualizer component for the educational website.
 *
 * This component provides an interactive visualization of how the Simulated Annealing
 * algorithm works on the Fantasy League problem.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, Button, Form, Row, Col, Badge, ProgressBar, Table } from 'react-bootstrap';

const SimulatedAnnealingVisualizer = () => {
  // State for visualization
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentStep, setCurrentStep] = useState(0);
  const [showExplanation, setShowExplanation] = useState(true);
  
  // State for algorithm parameters
  const [initialTemperature, setInitialTemperature] = useState(100);
  const [coolingRate, setCoolingRate] = useState(0.95);
  const [minTemperature, setMinTemperature] = useState(0.1);
  const [iterationsPerTemp, setIterationsPerTemp] = useState(10);
  
  // State for algorithm execution
  const [steps, setSteps] = useState([]);
  const [currentSolution, setCurrentSolution] = useState(null);
  const [bestSolution, setBestSolution] = useState(null);
  
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
    let bestFitness = currentFitness;
    let temperature = initialTemperature;
    let iteration = 0;
    
    while (temperature > minTemperature) {
      for (let i = 0; i < iterationsPerTemp; i++) {
        // Generate neighbor
        const neighborFitness = currentFitness + (Math.random() * 2 - 1) * Math.exp(-iteration / 50);
        
        // Calculate delta
        const delta = neighborFitness - currentFitness;
        
        // Determine if accepted
        let accepted = false;
        let acceptanceProbability = 0;
        
        if (delta <= 0) {
          // Better solution, always accept
          accepted = true;
          acceptanceProbability = 1.0;
        } else {
          // Worse solution, accept with probability
          acceptanceProbability = Math.exp(-delta / temperature);
          accepted = Math.random() < acceptanceProbability;
        }
        
        // Create step
        const step = {
          iteration: iteration,
          temperature: temperature,
          currentFitness: currentFitness,
          neighborFitness: neighborFitness,
          delta: delta,
          acceptanceProbability: acceptanceProbability,
          accepted: accepted,
          bestFitness: bestFitness
        };
        
        generatedSteps.push(step);
        
        // Update current fitness if accepted
        if (accepted) {
          currentFitness = neighborFitness;
          
          // Update best fitness if improved
          if (currentFitness < bestFitness) {
            bestFitness = currentFitness;
          }
        }
        
        iteration++;
      }
      
      // Cool down
      temperature *= coolingRate;
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
      
      if (step.accepted) {
        const newSolution = { ...currentSolution, fitness: step.neighborFitness };
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
      
      if (step.accepted) {
        const newSolution = { ...currentSolution, fitness: step.neighborFitness };
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
  }, [initialTemperature, coolingRate, minTemperature, iterationsPerTemp]);
  
  return (
    <div className="simulated-annealing-visualizer">
      <Card className="mb-4">
        <Card.Header>
          <h3>Simulated Annealing Algorithm Visualization</h3>
        </Card.Header>
        <Card.Body>
          <Row className="mb-4">
            <Col md={6}>
              <h4>Algorithm Parameters</h4>
              <Form>
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Initial Temperature:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={initialTemperature} 
                      onChange={(e) => setInitialTemperature(parseFloat(e.target.value))}
                      min={1}
                      max={1000}
                      step={1}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Cooling Rate:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={coolingRate} 
                      onChange={(e) => setCoolingRate(parseFloat(e.target.value))}
                      min={0.5}
                      max={0.99}
                      step={0.01}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Min Temperature:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={minTemperature} 
                      onChange={(e) => setMinTemperature(parseFloat(e.target.value))}
                      min={0.001}
                      max={10}
                      step={0.001}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Iterations Per Temp:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={iterationsPerTemp} 
                      onChange={(e) => setIterationsPerTemp(parseInt(e.target.value))}
                      min={1}
                      max={100}
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
                  <strong>Current Temperature:</strong> {
                    currentStep < steps.length ? 
                    steps[currentStep].temperature.toFixed(4) : 'N/A'
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
                <h4>Current Step Details</h4>
                <Table bordered hover>
                  <tbody>
                    <tr>
                      <th>Current Fitness:</th>
                      <td>{steps[currentStep].currentFitness.toFixed(4)}</td>
                      <th>Neighbor Fitness:</th>
                      <td>{steps[currentStep].neighborFitness.toFixed(4)}</td>
                    </tr>
                    <tr>
                      <th>Delta (ΔE):</th>
                      <td>{steps[currentStep].delta.toFixed(4)}</td>
                      <th>Acceptance Probability:</th>
                      <td>{steps[currentStep].acceptanceProbability.toFixed(4)}</td>
                    </tr>
                    <tr>
                      <th>Decision:</th>
                      <td colSpan={3}>
                        {steps[currentStep].delta <= 0 ? (
                          <div>
                            <Badge bg="success">Better Solution</Badge> - Always accepted
                          </div>
                        ) : (
                          <div>
                            <Badge bg="warning">Worse Solution</Badge> - Accepted with probability {steps[currentStep].acceptanceProbability.toFixed(4)}
                          </div>
                        )}
                      </td>
                    </tr>
                    <tr>
                      <th>Result:</th>
                      <td colSpan={3}>
                        <Badge variant={steps[currentStep].accepted ? "success" : "danger"}>
                          {steps[currentStep].accepted ? "Accepted" : "Rejected"}
                        </Badge>
                      </td>
                    </tr>
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
                    <h5>How Simulated Annealing Works</h5>
                    <p>
                      Simulated Annealing is inspired by the annealing process in metallurgy. It allows accepting worse solutions with a decreasing probability to escape local optima.
                    </p>
                    <ol>
                      <li>Start with a random initial solution and high temperature</li>
                      <li>Generate a random neighbor</li>
                      <li>If the neighbor is better (ΔE {'<'} 0), accept it</li>
                      <li>If the neighbor is worse (ΔE {'>'} 0), accept it with probability e<sup>-ΔE/T</sup></li>
                      <li>Decrease temperature according to cooling schedule</li>
                      <li>Repeat until temperature reaches minimum value</li>
                    </ol>
                    <h5>Key Metrics</h5>
                    <ul>
                      <li><strong>Acceptance Rate:</strong> Percentage of proposed moves that were accepted</li>
                      <li><strong>Worse Solutions Accepted:</strong> Number of worse solutions accepted</li>
                      <li><strong>Cooling Efficiency:</strong> How efficiently the temperature cooling schedule worked</li>
                      <li><strong>Temperature Impact:</strong> How temperature affected solution acceptance</li>
                    </ul>
                    <h5>Strengths and Weaknesses</h5>
                    <p><strong>Strengths:</strong> Can escape local optima, converges to global optimum with proper cooling schedule</p>
                    <p><strong>Weaknesses:</strong> Sensitive to parameter tuning, can be slow for large problems</p>
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

export default SimulatedAnnealingVisualizer;
