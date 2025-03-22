import numpy as np
import time
from typing import List, Optional

class RequestProfile:
    """Class to hold the arrival times of requests in a test."""
    def __init__(self, arrival_times: List[float]):
        self.arrival_times = arrival_times

class Metrics:
    """Placeholder for metrics class mentioned in your imports."""
    pass

class TraceGenerator:
    """
    Generates request arrival patterns for performance testing.
    
    This class creates various workload patterns:
    - constant: Steady workload at the base rate
    - spike: Base rate with sudden bursts of activity
    - sine: Sinusoidal pattern to simulate daily cycles
    - step: Stepped increases in workload
    """
    def __init__(
        self, 
        duration: float, 
        rate_pattern: str, 
        base_rate: float, 
        amplitude: float = 0.5, 
        noise: float = 0.1,
        spike_points: Optional[List[float]] = None
    ):
        """
        Initialize the trace generator with workload parameters.
        
        Args:
            duration: Test duration in seconds
            rate_pattern: Pattern type ('constant', 'spike', 'sine', or 'step')
            base_rate: Base request rate in requests per second
            amplitude: Variation amplitude (for non-constant patterns)
            noise: Random noise factor to add variation (0-1)
            spike_points: Optional list of points (0-1) where spikes should occur
        """
        self.duration = duration
        self.rate_pattern = rate_pattern
        self.base_rate = base_rate
        self.amplitude = amplitude
        self.noise = noise
        self.spike_points = spike_points or [0.5]  # Default spike in the middle
    
    def generate(self) -> RequestProfile:
        """
        Generate a profile of request arrival times based on the configured pattern.
        
        Returns:
            RequestProfile: Object containing the list of arrival times
        """
        # Start with current timestamp as base
        now = time.time()
        
        if self.rate_pattern == "constant":
            # For constant load, distribute evenly with some randomness
            # Expected request count given the rate and duration
            expected_count = int(self.base_rate * self.duration)
            
            # Generate arrival times with small random variations
            if self.noise > 0:
                # Use exponential distribution for realistic inter-arrival times
                intervals = np.random.exponential(1.0/self.base_rate, expected_count)
                arrival_times = np.cumsum(intervals)
                # Scale to fit within duration
                arrival_times = arrival_times * (self.duration / arrival_times[-1]) if arrival_times[-1] > 0 else arrival_times
            else:
                # Perfect uniform distribution if no noise
                arrival_times = np.linspace(0, self.duration, expected_count)
            
        elif self.rate_pattern == "spike":
            # Generate base load
            base_count = int(self.base_rate * self.duration)
            base_intervals = np.random.exponential(1.0/self.base_rate, base_count)
            base_times = np.cumsum(base_intervals)
            base_times = base_times[base_times < self.duration]
            
            # Generate spikes at specified points
            spike_times = []
            for spike_point in self.spike_points:
                spike_center = self.duration * spike_point
                spike_width = self.duration * 0.1  # 10% of duration
                spike_start = max(0, spike_center - spike_width/2)
                spike_end = min(self.duration, spike_center + spike_width/2)
                
                # Higher rate during spike
                spike_rate = self.base_rate + self.amplitude * self.base_rate
                spike_count = int(spike_rate * (spike_end - spike_start))
                
                spike_intervals = np.random.exponential(1.0/spike_rate, spike_count)
                these_spike_times = spike_start + np.cumsum(spike_intervals)
                these_spike_times = these_spike_times[these_spike_times < spike_end]
                
                spike_times.extend(these_spike_times)
            
            # Combine and sort all times
            arrival_times = np.sort(np.concatenate([base_times, spike_times]))
            
        elif self.rate_pattern == "sine":
            # For sinusoidal pattern (diurnal load)
            # Generate time points
            time_points = np.linspace(0, self.duration, 1000)
            
            # Calculate instantaneous rates at each time point
            # Sine wave oscillates between (base_rate - amplitude) and (base_rate + amplitude)
            rates = self.base_rate + self.amplitude * self.base_rate * np.sin(2 * np.pi * time_points / self.duration)
            
            # Add noise if specified
            if self.noise > 0:
                noise_values = np.random.normal(0, self.noise * self.base_rate, len(rates))
                rates = rates + noise_values
                # Ensure rates don't go negative
                rates = np.maximum(rates, 0.1)
            
            # Generate arrival times using non-homogeneous Poisson process
            arrival_times = []
            current_time = 0
            
            while current_time < self.duration:
                # Get current rate based on time (linear interpolation)
                idx = min(int(current_time / self.duration * len(time_points)), len(time_points) - 1)
                current_rate = rates[idx]
                
                # Generate next arrival time using exponential distribution
                interval = np.random.exponential(1.0 / current_rate) if current_rate > 0 else self.duration
                current_time += interval
                
                if current_time < self.duration:
                    arrival_times.append(current_time)
            
            arrival_times = np.array(arrival_times)
            
        elif self.rate_pattern == "step":
            # Step pattern with increasing load levels
            steps = 4  # Number of load steps
            arrival_times = []
            
            for i in range(steps):
                step_start = self.duration * i / steps
                step_end = self.duration * (i + 1) / steps
                step_duration = step_end - step_start
                
                # Increase rate with each step
                step_rate = self.base_rate * (1 + i * self.amplitude / (steps - 1))
                
                # Generate arrivals for this step
                step_count = int(step_rate * step_duration)
                step_intervals = np.random.exponential(1.0/step_rate, step_count)
                step_times = step_start + np.cumsum(step_intervals)
                step_times = step_times[step_times < step_end]
                
                arrival_times.extend(step_times)
            
            arrival_times = np.array(arrival_times)
            
        else:
            raise ValueError(f"Unknown rate pattern: {self.rate_pattern}")
        
        # Make times absolute by adding current time
        absolute_times = [now + t for t in arrival_times]
        
        return RequestProfile(absolute_times)