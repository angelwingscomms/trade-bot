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
from typing import Callable, List, Optional

@dataclass
class TickGenerator:
    """Generates synthetic tick data with configurable chaos + patterns"""
    
    base_price: float = 100.0
    spread: float = 0.1  # bid-ask spread
    spread_variance: float = 0.0  # variance in spread (0 = fixed spread)
    randomness: float = 5.0  # any positive float: multiplier for noise (unbounded)
    volatility: float = 0.5  # affects noise magnitude
    
    def _get_noise(self, num_ticks: int) -> np.ndarray:
        """Noise scaled by randomness level"""
        if self.randomness == 0:
            return np.zeros(num_ticks)
        
        noise = np.random.normal(0, self.volatility, num_ticks)
        return noise * self.randomness
    
    def trend_pattern(self, num_ticks: int, direction: float = 1.0) -> np.ndarray:
        """Linear trend + noise
        direction: 1.0 = up, -1.0 = down, 0.5 = weak up
        """
        trend = np.linspace(0, direction * 2.0, num_ticks)
        noise = self._get_noise(num_ticks)
        return self.base_price + trend + noise
    
    def mean_reversion_pattern(self, num_ticks: int, mean: Optional[float] = None) -> np.ndarray:
        """Oscillates around mean, real-world-like"""
        if mean is None:
            mean = self.base_price
        
        prices = np.full(num_ticks, mean)
        for i in range(1, num_ticks):
            reversion = 0.3 * (mean - prices[i-1])
            noise = np.random.normal(0, self.volatility) * self.randomness
            prices[i] = prices[i-1] + reversion + noise
        
        return prices
    
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
    
    def multi_scale_oscillation(self, num_ticks: int, frequencies: Optional[List[float]] = None) -> np.ndarray:
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
    
    def random_walk(self, num_ticks: int) -> np.ndarray:
        """Pure random walk - no pattern, just chaos. Hardest mode.
        Each step is random, no underlying trend or structure.
        """
        prices = [self.base_price]
        for _ in range(num_ticks - 1):
            step = np.random.normal(0, self.volatility) * self.randomness
            prices.append(prices[-1] + step)
        return np.array(prices)

    def generate_ticks(self, num_ticks: int, pattern_fn: Callable) -> np.ndarray:
        """Generate bid/ask prices from pattern function"""
        mid_prices = pattern_fn(num_ticks)
        
        if self.spread_variance > 0:
            spread_variation = np.random.normal(0, self.spread_variance, num_ticks)
            actual_spread = self.spread + spread_variation
            actual_spread = np.maximum(actual_spread, 0.001)
        else:
            actual_spread = self.spread
        
        bid = mid_prices - actual_spread / 2
        ask = mid_prices + actual_spread / 2
        
        return np.column_stack([bid, ask])

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tick data")
    parser.add_argument('-r', '--randomness', type=float, default=5.0, help='Randomness multiplier (any positive float, >100 warning)')
    parser.add_argument('-n', '--name', type=str, default='', help='Custom name for the output file')
    parser.add_argument('-c', '--count', type=int, default=540000, help='Number of ticks to generate')
    parser.add_argument('-p', '--pattern', type=str, default='mixed', choices=['trend', 'mean_reversion', 'reversal', 'oscillation', 'random', 'mixed'], help='Pattern to use')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-w', '--overwrite', action='store_true', help='Overwrite if file exists')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--interval', type=int, default=111, help='Time interval in ms between ticks')
    parser.add_argument('--precision', type=int, default=6, help='Decimal precision for float values')
    parser.add_argument('--volatility', type=float, default=0.5, help='Volatility parameter for noise generation')
    parser.add_argument('--spread-var', type=float, default=0.0, help='Spread variance (0 = fixed spread)')
    
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting synthetic tick data generation")

    # If no arguments provided, use defaults
    if len(sys.argv) == 1:
        current = datetime.datetime.now()
        mmss = current.strftime("%M%S")
        args.name = f"d{mmss}"
        logging.info(f"No arguments provided, using defaults: randomness={args.randomness}, count={args.count}, name={args.name}")

# If name still empty, generate d{mm}{ss}
    if not args.name:
        current = datetime.datetime.now()
        args.name = current.strftime("%M%S")
        args.name = f"d{args.name}"
    
    # Sanitize filename
    args.name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in args.name)
    if not args.name:
        args.name = 'unnamed'
    
    # Validate arguments
    if args.randomness < 0:
        logging.error(f"Randomness must be >= 0, got {args.randomness}")
        print(f"ERROR: Randomness must be >= 0, got {args.randomness}")
        sys.exit(1)
    if args.count <= 0:
        logging.error(f"Count must be > 0, got {args.count}")
        print(f"ERROR: Count must be > 0, got {args.count}")
        sys.exit(1)
    if args.interval <= 0:
        logging.error(f"Interval must be > 0, got {args.interval}")
        print(f"ERROR: Interval must be > 0, got {args.interval}")
        sys.exit(1)

    # Check if file already exists
    filename = f'data/synth/{args.name}.csv'
    # Always add .csv extension (in case user already included it)
    filename = filename.rstrip('.csv') + '.csv'
    if os.path.exists(filename) and not args.overwrite:
        logging.error(f"File already exists: {filename}")
        print(f"ERROR: File already exists: {filename}")
        sys.exit(1)

    gen = TickGenerator(randomness=args.randomness, volatility=args.volatility, spread_variance=args.spread_var)
    
    if args.randomness > 100:
        logging.warning(f"EXTREME RANDOMNESS: {args.randomness}x - data will be highly chaotic")
    
    if args.seed is not None:
        np.random.seed(args.seed)
        logging.info(f"Random seed set to {args.seed}")
    
    logging.info(f"Generating {args.count} ticks with pattern '{args.pattern}' and randomness {args.randomness}")

    if args.pattern == 'mixed':
        # Generate mixed patterns like scenario 4
        ticks = []
        patterns = [gen.trend_pattern, gen.mean_reversion_pattern, gen.reversal_pattern, gen.multi_scale_oscillation, gen.random_walk]
        segment_ticks = args.count // len(patterns)
        remainder = args.count % len(patterns)
        for i, pattern in enumerate(patterns):
            seg_count = segment_ticks + (1 if i < remainder else 0)
            logging.debug(f"Generating segment {i+1} with {seg_count} ticks")
            segment = gen.generate_ticks(seg_count, pattern)
            ticks.append(segment)
        all_ticks = np.vstack(ticks)
    else:
        pattern_map = {
            'trend': lambda: gen.trend_pattern(args.count),
            'mean_reversion': lambda: gen.mean_reversion_pattern(args.count),
            'reversal': lambda: gen.reversal_pattern(args.count),
            'oscillation': lambda: gen.multi_scale_oscillation(args.count),
            'random': lambda: gen.random_walk(args.count)
        }
        all_ticks = gen.generate_ticks(args.count, pattern_map[args.pattern])

    logging.info(f"Generated {len(all_ticks)} ticks")

    # Add timestamps
    start_time = np.int64(time.time() * 1000)
    times = (np.arange(len(all_ticks), dtype=np.int64) * args.interval + start_time)
    all_ticks = np.column_stack([times, all_ticks])

    # Ensure directory exists
    os.makedirs('data/synth', exist_ok=True)

    # Save to CSV
    logging.info(f"Saving {len(all_ticks)} ticks to {filename}")
    fmt = f'%.{args.precision}f'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_msc', 'bid', 'ask'])
        for row in all_ticks:
            writer.writerow([row[0], fmt % row[1], fmt % row[2]])

    logging.info("Data generation and saving completed")
    logging.info(f"Randomness level used: {args.randomness}")
    print(f"Generated {len(all_ticks)} ticks and saved to {filename}")

if __name__ == '__main__':
    main()