#!/usr/bin/env python3
import argparse
import numpy as np
import csv
import os
import time
import datetime
import sys
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

class Randomness(Enum):
    """Randomness levels: 0=deterministic, 10=pure random"""
    NONE = 0
    LOW = 2
    MEDIUM = 5
    HIGH = 8
    CHAOS = 10

@dataclass
class TickGenerator:
    """Generates synthetic tick data with configurable chaos + patterns"""
    
    base_price: float = 100.0
    spread: float = 0.1  # bid-ask spread
    randomness: Randomness = Randomness.MEDIUM
    volatility: float = 0.5  # affects noise magnitude
    
    def _get_noise(self, num_ticks: int) -> np.ndarray:
        """Noise scaled by randomness level"""
        if self.randomness.value == 0:
            return np.zeros(num_ticks)
        
        noise = np.random.normal(0, self.volatility, num_ticks)
        return noise * (self.randomness.value / 10.0)
    
    def trend_pattern(self, num_ticks: int, direction: float = 1.0) -> np.ndarray:
        """Linear trend + noise
        direction: 1.0 = up, -1.0 = down, 0.5 = weak up
        """
        trend = np.linspace(0, direction * 2.0, num_ticks)
        noise = self._get_noise(num_ticks)
        return self.base_price + trend + noise
    
    def mean_reversion_pattern(self, num_ticks: int, mean: float = None) -> np.ndarray:
        """Oscillates around mean, real-world-like"""
        if mean is None:
            mean = self.base_price
        
        prices = [mean]
        for _ in range(num_ticks - 1):
            # Pull toward mean 30% + random walk
            reversion = 0.3 * (mean - prices[-1])
            random_step = np.random.normal(0, self.volatility)
            noise = random_step * (self.randomness.value / 10.0)
            prices.append(prices[-1] + reversion + noise)
        
        return np.array(prices)
    
    def reversal_pattern(self, num_ticks: int, pivot_point: float = 0.5) -> np.ndarray:
        """Reversal: up then down (or vice versa)
        pivot_point: where (0-1) the reversal happens
        """
        pivot_idx = int(num_ticks * pivot_point)
        
        up_trend = np.linspace(0, 3.0, pivot_idx)
        down_trend = np.linspace(3.0, 0, num_ticks - pivot_idx)
        
        trend = np.concatenate([up_trend, down_trend])
        noise = self._get_noise(num_ticks)
        return self.base_price + trend + noise
    
    def multi_scale_oscillation(self, num_ticks: int, frequencies: list[float] = None) -> np.ndarray:
        """Multiple sine waves at different scales (realistic complexity)
        frequencies: list of oscillation frequencies
        """
        if frequencies is None:
            frequencies = [0.02, 0.005, 0.001]  # Multiple timeframes
        
        t = np.arange(num_ticks)
        price = self.base_price
        
        for freq in frequencies:
            price = price + np.sin(2 * np.pi * freq * t) * (2.0 / len(frequencies))
        
        noise = self._get_noise(num_ticks)
        return price + noise

    def generate_ticks(self, num_ticks: int, pattern_fn: Callable) -> np.ndarray:
        """Generate bid/ask prices from pattern function"""
        mid_prices = pattern_fn(num_ticks)
        
        bid = mid_prices - self.spread / 2
        ask = mid_prices + self.spread / 2
        
        return np.column_stack([bid, ask])

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tick data")
    parser.add_argument('-r', '--randomness', type=int, default=5, choices=range(11), help='Randomness level (0-10)')
    parser.add_argument('-n', '--name', type=str, default='default', help='Custom name for the output file')
    parser.add_argument('-c', '--count', type=int, default=1500, help='Number of ticks to generate')
    parser.add_argument('-p', '--pattern', type=str, default='mixed', choices=['trend', 'mean_reversion', 'reversal', 'oscillation', 'mixed'], help='Pattern to use')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting synthetic tick data generation")

    # If no arguments provided, use defaults
    if len(sys.argv) == 1:
        current = datetime.datetime.now()
        mmss = current.strftime("%M%S")
        args.randomness = 0
        args.count = 14400000
        args.name = f"default{mmss}"
        logging.info(f"No arguments provided, using defaults: randomness={args.randomness}, count={args.count}, name={args.name}")

    randomness = Randomness(args.randomness)
    gen = TickGenerator(randomness=randomness)
    
    logging.info(f"Generating {args.count} ticks with pattern '{args.pattern}' and randomness {args.randomness}")

    if args.pattern == 'mixed':
        # Generate mixed patterns like scenario 4
        ticks = []
        patterns = [gen.trend_pattern, gen.mean_reversion_pattern, gen.reversal_pattern]
        segment_ticks = args.count // len(patterns)
        for i, pattern in enumerate(patterns):
            logging.debug(f"Generating segment {i+1} with {segment_ticks} ticks")
            segment = gen.generate_ticks(segment_ticks, pattern)
            ticks.append(segment)
        all_ticks = np.vstack(ticks)
    else:
        pattern_map = {
            'trend': lambda: gen.trend_pattern(args.count),
            'mean_reversion': lambda: gen.mean_reversion_pattern(args.count),
            'reversal': lambda: gen.reversal_pattern(args.count),
            'oscillation': lambda: gen.multi_scale_oscillation(args.count)
        }
        mid_prices = pattern_map[args.pattern]()
        all_ticks = np.column_stack([mid_prices - gen.spread/2, mid_prices + gen.spread/2])

    logging.info(f"Generated {len(all_ticks)} ticks")

    # Add timestamps
    start_time = int(time.time() * 1000)
    times = np.arange(len(all_ticks)) * 100 + start_time  # 100ms intervals
    all_ticks = np.column_stack([times, all_ticks])

    # Ensure directory exists
    os.makedirs('data/synth', exist_ok=True)

    # Save to CSV
    filename = f'data/synth/{args.name}.csv'
    logging.info(f"Saving {len(all_ticks)} ticks to {filename}")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_msc', 'bid', 'ask'])
        writer.writerows(all_ticks)

    logging.info("Data generation and saving completed")
    print(f"Generated {len(all_ticks)} ticks and saved to {filename}")

if __name__ == '__main__':
    main()
