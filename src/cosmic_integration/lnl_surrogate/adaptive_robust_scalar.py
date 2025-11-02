import logging
import numpy as np
from typing import Optional, List, Callable
import warnings
import json
from tqdm.auto import tqdm


class AdaptiveRobustScaler:
    """
    Robust scaler that adapts statistics as new data arrives and can reject bad points.
    Uses percentiles and median for robustness to outliers with integrated rejection logic.

    Idea from:
    https://www.sciencedirect.com/science/article/abs/pii/S0142061521006402
    https://www.sciencedirect.com/science/article/abs/pii/S0167947322001177
    """

    def __init__(self,
                 percentile_range: tuple = (25, 75),
                 soft_clipping: bool = True,
                 clip_factor: float = 2.0,
                 min_samples_for_update: int = 10,
                 reject_bad_points: bool = False,
                 adaptive_rejection_threshold: bool = True):
        self.percentile_range = percentile_range
        self.soft_clipping = soft_clipping
        self.clip_factor = clip_factor
        self.min_samples_for_update = min_samples_for_update
        self.reject_bad_points = reject_bad_points
        self.adaptive_rejection_threshold = adaptive_rejection_threshold

        # Statistics
        self.median = 0.0
        self.scale = 1.0
        self.reference_value = 0.0
        self.n_samples = 0

        # Store recent values for rolling updates
        self.recent_values = []
        self.update_history = []

        # Rejection system (only used if reject_bad_points=True)
        self.rejection_tracker = {
            'total_evaluations': 0,
            'rejected_count': 0,
            'accepted_count': 0,
            'rejection_rate': 0.0,
            'acceptance_rate': 0.0,
            'threshold_history': []
        }
        self.rejection_threshold = float('-inf')  # No rejection by default

    def initialize_with_data(self, initial_lnls: np.ndarray):
        """Initialize the scaler with initial data"""
        # Convert to numpy array if needed
        initial_lnls = np.array(initial_lnls)

        # Initialize reference
        for lnl in initial_lnls:
            self.update_reference(lnl)

        # Initialize statistics
        self.update_stats(initial_lnls)

        # Initialize rejection if enabled
        if self.reject_bad_points and len(initial_lnls) > 0:
            self._init_rejection_threshold(initial_lnls)

    def _init_rejection_threshold(self, initial_lnls: np.ndarray):
        """Initialize rejection threshold based on initial data with focus on high-likelihood regions"""
        sorted_lnls = np.sort(initial_lnls)

        if len(sorted_lnls) < 10:
            # Too few points, be conservative
            self.rejection_threshold = np.min(sorted_lnls) - 5.0
        else:
            # Focus on upper percentiles for high-likelihood exploration
            q90, q75, q50 = np.percentile(sorted_lnls, [90, 75, 50])

            # Use median + some fraction toward q90 instead of q75 - 3IQR
            # This focuses on the better half and encourages exploration of high-likelihood regions
            self.rejection_threshold = q50 + 0.5 * (q90 - q50)  # 75th percentile equivalent

        # Ensure we don't reject everything - maintain some diversity
        min_threshold_range = 50.0  # At least 50 units below the best
        best_initial = np.max(initial_lnls) if len(initial_lnls) > 0 else 0
        self.rejection_threshold = min(self.rejection_threshold, best_initial - min_threshold_range)

        self.rejection_tracker['threshold_history'].append(self.rejection_threshold)

    def update_reference(self, new_value: float):
        """Update the reference point (running maximum for log-likelihood)"""
        if self.n_samples == 0:
            self.reference_value = new_value
        else:
            self.reference_value = max(self.reference_value, new_value)

    def update_stats(self, raw_values: np.ndarray):
        """Update robust statistics from raw values"""
        if len(raw_values) == 0:
            return

        # Shift by current reference
        shifted_values = raw_values - self.reference_value

        if self.n_samples == 0:
            # First update - initialize
            self.median = np.median(shifted_values)
            p_low, p_high = np.percentile(shifted_values, self.percentile_range)
            self.scale = max(p_high - p_low, 1e-10)  # Prevent division by zero
        else:
            # Combine with existing values for rolling statistics
            all_values = np.concatenate([
                np.array(self.recent_values),
                shifted_values
            ])

            # Keep only recent values (last 1000) for efficiency
            if len(all_values) > 1000:
                all_values = all_values[-1000:]

            # Update robust statistics
            self.median = np.median(all_values)
            p_low, p_high = np.percentile(all_values, self.percentile_range)
            self.scale = max(p_high - p_low, 1e-10)

        # Update sample count and recent values
        self.n_samples += len(raw_values)
        self.recent_values.extend(shifted_values.tolist())

        # Keep recent_values manageable
        if len(self.recent_values) > 500:
            self.recent_values = self.recent_values[-500:]

        # Track update history for diagnostics
        self.update_history.append({
            'n_samples': self.n_samples,
            'median': self.median,
            'scale': self.scale,
            'reference': self.reference_value
        })

    def should_reject(self, lnl_value: float, eval_count: int = 0):
        """Determine if a log-likelihood value should be rejected"""
        if not self.reject_bad_points:
            return {'reject': False}

        if lnl_value < self.rejection_threshold:
            self.rejection_tracker['rejected_count'] += 1
            self.rejection_tracker['total_evaluations'] = eval_count
            if eval_count > 0:
                self.rejection_tracker['rejection_rate'] = self.rejection_tracker['rejected_count'] / eval_count

            # Adapt threshold if enabled
            if self.adaptive_rejection_threshold and eval_count % 100 == 0:
                self._adapt_rejection_threshold()

            return {
                'reject': True,
                'threshold': self.rejection_threshold,
                'penalty_value': self.rejection_threshold - 10.0
            }

        self.rejection_tracker['accepted_count'] += 1
        self.rejection_tracker['total_evaluations'] = eval_count
        if eval_count > 0:
            self.rejection_tracker['acceptance_rate'] = self.rejection_tracker['accepted_count'] / eval_count

        return {'reject': False}

    def _adapt_rejection_threshold(self):
        """Adapt rejection threshold based on recent rejection rates - biased toward higher rejection"""
        if self.rejection_tracker['total_evaluations'] == 0:
            return

        rej_rate = self.rejection_tracker['rejection_rate']

        # Adjust threshold based on rejection rate - optimized for high-likelihood exploration
        if rej_rate > 0.6:  # Changed from 0.7
            # Too many rejections - threshold might be too loose
            self.rejection_threshold -= 0.5  # Reduced from 2.0 (more conservative adjustments)
        elif rej_rate < 0.4:  # Changed from 0.3
            # Moderate rejections - still want higher rejection, so strengthen threshold
            self.rejection_threshold += 1.5  # Increased from 1.0 (more aggressive tightening)
        elif rej_rate < 0.2:
            # Very low rejection - aggressively increase to focus on high-likelihood regions
            self.rejection_threshold += 2.0

        # Clamp threshold to reasonable bounds
        min_threshold = (self.reference_value - 200)  # Increased minimum range
        max_threshold = self.reference_value - 10  # Ensure some rejections are always possible
        self.rejection_threshold = np.clip(self.rejection_threshold, min_threshold, max_threshold)

        self.rejection_tracker['threshold_history'].append(self.rejection_threshold)

    def transform(self, raw_value: float) -> float:
        """Transform a single raw value"""
        if self.scale == 0:
            return 0.0

        # Shift by reference and standardize
        shifted = raw_value - self.reference_value
        standardized = (shifted - self.median) / self.scale

        if self.soft_clipping:
            # Soft clipping using tanh to prevent extreme values
            return np.tanh(standardized / self.clip_factor) * self.clip_factor
        else:
            return standardized

    def inverse_transform(self, transformed_value: float) -> float:
        """Inverse transform back to original space"""
        if self.soft_clipping:
            # Inverse tanh
            clipped_std = transformed_value / self.clip_factor
            # Clamp to prevent numerical issues with atanh
            clipped_std = np.clip(clipped_std, -0.999, 0.999)
            standardized = np.arctanh(clipped_std) * self.clip_factor
        else:
            standardized = transformed_value

        # Inverse standardization
        shifted = standardized * self.scale + self.median
        return shifted + self.reference_value

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about the scaler and rejection system"""
        diagnostics = {
            'n_samples': self.n_samples,
            'current_reference': self.reference_value,
            'current_median': self.median,
            'current_scale': self.scale,
            'effective_range': (self.median - 2 * self.scale, self.median + 2 * self.scale),
            'update_history': self.update_history[-10:]  # Last 10 updates
        }

        if self.reject_bad_points:
            diagnostics.update({
                'rejection_enabled': True,
                'current_threshold': self.rejection_threshold,
                'rejection_stats': self.rejection_tracker.copy()
            })

        return diagnostics

    def save(self, outdir):
        """Save the scaler state"""
        fname = f"{outdir}/scaler.json"
        state = {
            'percentile_range': self.percentile_range,
            'soft_clipping': self.soft_clipping,
            'clip_factor': self.clip_factor,
            'min_samples_for_update': self.min_samples_for_update,
            'reject_bad_points': self.reject_bad_points,
            'adaptive_rejection_threshold': self.adaptive_rejection_threshold,
            'median': float(self.median),
            'scale': float(self.scale),
            'reference_value': float(self.reference_value),
            'n_samples': int(self.n_samples),
            'recent_values': [float(v) for v in self.recent_values],
            'update_history': self.update_history,
            'rejection_tracker': self.rejection_tracker,
            'rejection_threshold': float(self.rejection_threshold),
        }
        with open(fname, 'w') as f:
            json.dump(state, f, indent=2)

        diagnostics = self.get_diagnostics()
        logger = logging.getLogger(__name__)
        logger.info("Scaler diagnostics saved:")
        logger.info(f"  Reference: {diagnostics['current_reference']:.2f}")
        logger.info(f"  Median: {diagnostics['current_median']:.3f}")
        logger.info(f"  Scale: {diagnostics['current_scale']:.3f}")
        if self.reject_bad_points:
            logger.info(f"  Rejection threshold: {diagnostics.get('current_threshold', 'N/A'):.2f}")

    @classmethod
    def load(cls, outdir):
        """Load scaler from saved state"""
        fname = f"{outdir}/scaler.json"
        with open(fname, 'r') as f:
            state = json.load(f)
        scaler = cls(
            percentile_range=tuple(state['percentile_range']),
            soft_clipping=state['soft_clipping'],
            clip_factor=state['clip_factor'],
            min_samples_for_update=state['min_samples_for_update'],
            reject_bad_points=state.get('reject_bad_points', False),
            adaptive_rejection_threshold=state.get('adaptive_rejection_threshold', True)
        )
        scaler.median = state['median']
        scaler.scale = state['scale']
        scaler.reference_value = state['reference_value']
        scaler.n_samples = state['n_samples']
        scaler.recent_values = state['recent_values']
        scaler.update_history = state['update_history']
        scaler.rejection_tracker = state.get('rejection_tracker', {})
        scaler.rejection_threshold = state.get('rejection_threshold', float('-inf'))
        return scaler


def robust_neg_lnl_computer_factory(lnl_computer, initial_lnls: np.ndarray, reject_bad_points: bool = True) -> Callable:
    """
    Factory function to create a robust negative log-likelihood computer
    with adaptive reference updates and robust standardization.

    Can optionally reject bad points to focus acquisition on promising regions.
    """
    # Initialize the scaler with initial data and rejection settings
    scaler = AdaptiveRobustScaler(
        percentile_range=(25, 75),
        soft_clipping=True,
        clip_factor=3.0,  # Maps most values to [-3, 3]
        min_samples_for_update=20,
        reject_bad_points=reject_bad_points
    )

    # Initialize the scaler with initial data
    scaler.initialize_with_data(initial_lnls)

    # Counter for periodic updates
    eval_counter = {'count': 0, 'recent_values': []}

    def robust_neg_lnl_computer(*params):
        """
        Robust negative log-likelihood computer with optional bad point rejection
        """
        params = np.array(params).flatten()

        try:
            # Compute raw log-likelihood
            raw_lnl = lnl_computer(*params)

            # Update reference point (always done)
            scaler.update_reference(raw_lnl)

            # Check if point should be rejected
            rejection_result = scaler.should_reject(raw_lnl, eval_counter['count'] + 1)

            if rejection_result['reject']:
                eval_counter['count'] += 1  # Still count as evaluation

                # Update scaler stats for consistency
                if len(eval_counter['recent_values']) > 0:
                    scaler.update_stats(np.array(eval_counter['recent_values']))
                    eval_counter['recent_values'] = []

                # Log rejections occasionally
                if eval_counter['count'] % 50 == 0:
                    logger = logging.getLogger(__name__)
                    logger.warning("Rejecting point with lnL={:.1f} < {:.1f} "
                             "(rejection rate: {:.0f}%)".format(
                                 raw_lnl,
                                 rejection_result['threshold'],
                                 scaler.rejection_tracker.get('rejection_rate', 0) * 100
                             ))

                # Return penalty value
                penalty_raw = rejection_result['penalty_value']
                penalty_value = scaler.transform(penalty_raw)
                return -penalty_value

            # Accept this point - normal processing
            # Store for batch updates
            eval_counter['recent_values'].append(raw_lnl)
            eval_counter['count'] += 1

            # Periodic batch updates of statistics and scaling
            if eval_counter['count'] % scaler.min_samples_for_update == 0:
                if len(eval_counter['recent_values']) > 0:
                    scaler.update_stats(np.array(eval_counter['recent_values']))
                    eval_counter['recent_values'] = []  # Clear the buffer

                    # Print diagnostics occasionally, including rejection stats if active
                    if eval_counter['count'] % 100 == 0:
                        diagnostics = scaler.get_diagnostics()
                        logger = logging.getLogger(__name__)
                        logger.info(f"Scaler update at evaluation {eval_counter['count']}:")
                        logger.info(f"  Reference: {diagnostics['current_reference']:.2f}")
                        logger.info(f"  Median: {diagnostics['current_median']:.3f}")
                        logger.info(f"  Scale: {diagnostics['current_scale']:.3f}")

                        if scaler.reject_bad_points:
                            rej_stats = diagnostics.get('rejection_stats', {})
                            logger.info(f"  Rejection rate: {rej_stats.get('rejection_rate', 0)*100:.1f}%")
                            logger.info(f"  Acceptance rate: {rej_stats.get('acceptance_rate', 0)*100:.1f}%")

            # Transform the value
            transformed = scaler.transform(raw_lnl)

            # Return negative for minimization
            return -transformed

        except Exception as e:
            # Handle evaluation failures gracefully
            warnings.warn(f"LnL evaluation failed: {e}")
            # Return a penalty value
            penalty = scaler.transform(scaler.reference_value - 1000)
            eval_counter['count'] += 1  # Count as evaluation even on failure
            return -penalty

    # Attach components for external access
    robust_neg_lnl_computer.scaler = scaler
    robust_neg_lnl_computer.eval_counter = eval_counter

    return robust_neg_lnl_computer
