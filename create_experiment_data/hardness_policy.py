"""
HardnessPolicy: Centralized difficulty management for XAI experiments

This module provides a unified approach to classifying and stratifying instances
by their prediction difficulty/hardness based on model uncertainty (distance from 0.5).
"""

import logging
import gin
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)


@gin.configurable
class HardnessPolicy:
    """
    Centralized policy for managing instance difficulty/hardness classification.
    
    Hardness is determined by model uncertainty - instances closer to the decision
    boundary (probability near 0.5) are considered harder than those with confident
    predictions (probability near 0.0 or 1.0).
    
    This policy ensures consistent hardness definitions and target distributions
    across both training instance selection (DiverseInstances) and test instance
    generation (DemandDrivenTestInstanceManager).
    """
    
    def __init__(self,
                 hard_range: Tuple[float, float] = (0.4, 0.6),
                 medium_range: List[Tuple[float, float]] = [(0.3, 0.4), (0.6, 0.7)],
                 easy_range: List[Tuple[float, float]] = [(0.0, 0.3), (0.7, 1.0)],
                 target_mix: Dict[str, float] = None,
                 enabled: bool = True):
        """
        Initialize the hardness policy.
        
        Args:
            hard_range: Range of probabilities considered "hard" (near decision boundary)
            medium_range: List of ranges considered "medium" difficulty
            easy_range: List of ranges considered "easy" (confident predictions)
            target_mix: Target distribution {"hard": 0.6, "medium": 0.3, "easy": 0.1}
            enabled: Whether hardness stratification is enabled
        """
        self.hard_range = hard_range
        self.medium_range = medium_range
        self.easy_range = easy_range
        self.target_mix = target_mix or {"hard": 0.6, "medium": 0.3, "easy": 0.1}
        self.enabled = enabled
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"HardnessPolicy initialized:")
        logger.info(f"  Hard range: {self.hard_range}")
        logger.info(f"  Medium ranges: {self.medium_range}")
        logger.info(f"  Easy ranges: {self.easy_range}")
        logger.info(f"  Target mix: {self.target_mix}")
        logger.info(f"  Enabled: {self.enabled}")
    
    def _validate_configuration(self):
        """Validate hardness ranges and target mix."""
        # Check that target mix sums to 1.0
        total_mix = sum(self.target_mix.values())
        if abs(total_mix - 1.0) > 0.001:
            raise ValueError(f"Target mix must sum to 1.0, got {total_mix}: {self.target_mix}")
        
        # Check that all required categories are present
        required_categories = {"hard", "medium", "easy"}
        if set(self.target_mix.keys()) != required_categories:
            raise ValueError(f"Target mix must contain exactly {required_categories}, got {set(self.target_mix.keys())}")
        
        # Check ranges are within [0, 1]
        all_ranges = [self.hard_range] + self.medium_range + self.easy_range
        for range_tuple in all_ranges:
            if not (0.0 <= range_tuple[0] <= range_tuple[1] <= 1.0):
                raise ValueError(f"Invalid range {range_tuple}: must be within [0, 1] and properly ordered")
    
    def get_instance_probability(self, model, instance: pd.DataFrame) -> float:
        """
        Get model prediction probability for a single instance.
        
        Args:
            model: Trained model with predict_proba method
            instance: Single instance as DataFrame
            
        Returns:
            Probability of positive class (class 1)
        """
        try:
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(instance)
                # Return probability of class 1 (positive class)
                if probabilities.shape[1] > 1:
                    return probabilities[0, 1]
                else:
                    # Binary case with single probability
                    return probabilities[0, 0]
            else:
                # Fallback to regular prediction if no predict_proba
                prediction = model.predict(instance)[0]
                return float(prediction)
        except Exception as e:
            logger.warning(f"Error getting instance probability: {e}")
            return 0.5  # Default to boundary case
    
    def classify_hardness(self, probability: float) -> str:
        """
        Classify a probability as hard, medium, or easy.
        
        Args:
            probability: Model prediction probability for positive class
            
        Returns:
            "hard", "medium", or "easy"
        """
        # Check hard range first (most specific)
        if self.hard_range[0] <= probability <= self.hard_range[1]:
            return "hard"
        
        # Check medium ranges
        for medium_min, medium_max in self.medium_range:
            if medium_min <= probability <= medium_max:
                return "medium"
        
        # Check easy ranges
        for easy_min, easy_max in self.easy_range:
            if easy_min <= probability <= easy_max:
                return "easy"
        
        # Fallback (shouldn't happen with proper configuration)
        logger.warning(f"Probability {probability} doesn't fit any hardness category, defaulting to 'medium'")
        return "medium"
    
    def get_instance_hardness(self, model, instance: pd.DataFrame) -> str:
        """
        Get hardness classification for a single instance.
        
        Args:
            model: Trained model
            instance: Single instance as DataFrame
            
        Returns:
            "hard", "medium", or "easy"
        """
        if not self.enabled:
            return "medium"  # Neutral category when disabled
        
        probability = self.get_instance_probability(model, instance)
        return self.classify_hardness(probability)
    
    def get_hardness_distribution(self, model, instances: List[int], 
                                data: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Group instances by their hardness classification.
        
        Args:
            model: Trained model
            instances: List of instance IDs
            data: DataFrame containing the instances
            
        Returns:
            Dictionary mapping hardness level to list of instance IDs
        """
        if not self.enabled:
            return {"medium": instances}  # All instances in neutral category
        
        hardness_groups = {"hard": [], "medium": [], "easy": []}
        
        for instance_id in instances:
            try:
                instance = data.loc[[instance_id]]
                hardness = self.get_instance_hardness(model, instance)
                hardness_groups[hardness].append(instance_id)
            except Exception as e:
                logger.warning(f"Error classifying hardness for instance {instance_id}: {e}")
                # Default to medium for problematic instances
                hardness_groups["medium"].append(instance_id)
        
        return hardness_groups
    
    def stratify_instances(self, model, instances: List[int], data: pd.DataFrame,
                          target_count: int) -> List[int]:
        """
        Select instances according to target hardness distribution.
        
        Args:
            model: Trained model
            instances: Pool of candidate instance IDs
            data: DataFrame containing the instances
            target_count: Total number of instances to select
            
        Returns:
            List of selected instance IDs with target hardness distribution
        """
        if not self.enabled or target_count <= 0:
            return instances[:target_count]
        
        # Group instances by hardness
        hardness_groups = self.get_hardness_distribution(model, instances, data)
        
        # Calculate target counts for each hardness level
        target_counts = {}
        remaining_count = target_count
        
        for hardness, ratio in self.target_mix.items():
            if hardness == "easy":  # Handle easy last to absorb rounding errors
                target_counts[hardness] = remaining_count
            else:
                count = int(target_count * ratio)
                target_counts[hardness] = count
                remaining_count -= count
        
        # Select instances from each hardness group
        selected_instances = []
        shortfall_instances = []
        
        for hardness, target_needed in target_counts.items():
            available = hardness_groups[hardness]
            
            if len(available) >= target_needed:
                # Sufficient instances available
                selected_instances.extend(available[:target_needed])
            else:
                # Not enough instances, take all available and track shortfall
                selected_instances.extend(available)
                shortfall_needed = target_needed - len(available)
                
                # Collect instances from other categories to fill shortfall
                other_hardness_levels = [h for h in hardness_groups.keys() if h != hardness]
                for other_hardness in other_hardness_levels:
                    other_available = [id for id in hardness_groups[other_hardness] 
                                     if id not in selected_instances]
                    take_count = min(shortfall_needed, len(other_available))
                    shortfall_instances.extend(other_available[:take_count])
                    shortfall_needed -= take_count
                    if shortfall_needed <= 0:
                        break
                
                if shortfall_needed > 0:
                    logger.warning(f"Could not fully satisfy hardness target for '{hardness}': "
                                 f"needed {target_needed}, got {len(available)} + {len(shortfall_instances)} fallback")
        
        # Combine selected and shortfall instances
        final_selection = selected_instances + shortfall_instances
        
        # Ensure we don't exceed target count
        final_selection = final_selection[:target_count]
        
        # Log results
        self._log_stratification_results(model, final_selection, data, target_count, target_counts)
        
        return final_selection
    
    def _log_stratification_results(self, model, selected_instances: List[int], 
                                  data: pd.DataFrame, target_count: int,
                                  target_counts: Dict[str, int]):
        """Log the results of hardness stratification."""
        if not selected_instances:
            return
        
        # Get actual distribution
        actual_distribution = self.get_hardness_distribution(model, selected_instances, data)
        actual_counts = {h: len(ids) for h, ids in actual_distribution.items()}
        
        logger.info(f"Hardness stratification results (target: {target_count}):")
        for hardness in ["hard", "medium", "easy"]:
            target = target_counts.get(hardness, 0)
            actual = actual_counts.get(hardness, 0)
            ratio = actual / len(selected_instances) if selected_instances else 0
            logger.info(f"  {hardness.capitalize()}: {actual}/{len(selected_instances)} "
                       f"(target: {target}, ratio: {ratio:.3f})")
    
    def generate_hardness_report(self, model, instances: List[int], 
                               data: pd.DataFrame, 
                               title: str = "Hardness Distribution") -> Dict:
        """
        Generate a comprehensive hardness distribution report.
        
        Args:
            model: Trained model
            instances: List of instance IDs to analyze
            data: DataFrame containing the instances
            title: Title for the report
            
        Returns:
            Dictionary containing hardness analysis
        """
        if not self.enabled:
            return {
                "title": title,
                "enabled": False,
                "message": "Hardness policy is disabled"
            }
        
        # Get hardness distribution
        hardness_groups = self.get_hardness_distribution(model, instances, data)
        
        # Calculate statistics
        total_instances = len(instances)
        hardness_stats = {}
        
        for hardness, instance_ids in hardness_groups.items():
            count = len(instance_ids)
            ratio = count / total_instances if total_instances > 0 else 0
            target_ratio = self.target_mix.get(hardness, 0)
            
            hardness_stats[hardness] = {
                "count": count,
                "ratio": ratio,
                "target_ratio": target_ratio,
                "difference": ratio - target_ratio
            }
        
        # Calculate probabilities for a sample of instances
        sample_probabilities = []
        sample_size = min(10, len(instances))
        sample_instances = instances[:sample_size]
        
        for instance_id in sample_instances:
            try:
                instance = data.loc[[instance_id]]
                probability = self.get_instance_probability(model, instance)
                hardness = self.classify_hardness(probability)
                sample_probabilities.append({
                    "instance_id": instance_id,
                    "probability": probability,
                    "hardness": hardness
                })
            except Exception as e:
                logger.warning(f"Error analyzing instance {instance_id}: {e}")
        
        return {
            "title": title,
            "enabled": True,
            "total_instances": total_instances,
            "hardness_stats": hardness_stats,
            "sample_probabilities": sample_probabilities,
            "target_mix": self.target_mix,
            "configuration": {
                "hard_range": self.hard_range,
                "medium_range": self.medium_range,
                "easy_range": self.easy_range
            }
        }
    
    def print_hardness_report(self, model, instances: List[int], 
                            data: pd.DataFrame, 
                            title: str = "Hardness Distribution"):
        """Print a formatted hardness distribution report."""
        report = self.generate_hardness_report(model, instances, data, title)
        
        print("=" * 50)
        print(f"HARDNESS POLICY REPORT: {report['title']}")
        print("=" * 50)
        
        if not report['enabled']:
            print(report['message'])
            print("=" * 50)
            return
        
        print(f"Total instances: {report['total_instances']}")
        print()
        
        # Print hardness distribution
        for hardness in ["hard", "medium", "easy"]:
            stats = report['hardness_stats'].get(hardness, {})
            count = stats.get('count', 0)
            ratio = stats.get('ratio', 0)
            target_ratio = stats.get('target_ratio', 0)
            difference = stats.get('difference', 0)
            
            status = "✓" if abs(difference) <= 0.1 else "✗"
            print(f"{hardness.capitalize()}: {count} ({ratio:.3f}) "
                  f"[target: {target_ratio:.3f}, diff: {difference:+.3f}] {status}")
        
        print()
        print("Sample instance probabilities:")
        for sample in report['sample_probabilities'][:5]:  # Show first 5
            print(f"  ID {sample['instance_id']}: {sample['probability']:.3f} -> {sample['hardness']}")
        
        print("=" * 50)