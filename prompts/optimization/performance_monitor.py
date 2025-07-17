"""
Tuthand Performance Monitoring System
Real-time monitoring and optimization for prompt performance
"""

import time
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import statistics

@dataclass
class InteractionMetrics:
    """Comprehensive interaction metrics"""
    interaction_id: str
    timestamp: datetime
    user_type: str
    strategy: str
    query: str
    response: str
    confidence_score: float
    trust_level: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    response_time: float
    cost_estimate: float
    user_satisfaction: Optional[float] = None
    escalated: bool = False
    cache_hit: bool = False

class PerformanceDatabase:
    """SQLite database for storing performance metrics"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                timestamp TEXT,
                user_type TEXT,
                strategy TEXT,
                query TEXT,
                response TEXT,
                confidence_score REAL,
                trust_level TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                response_time REAL,
                cost_estimate REAL,
                user_satisfaction REAL,
                escalated INTEGER,
                cache_hit INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                message TEXT,
                severity TEXT,
                resolved INTEGER DEFAULT 0
            )
        ''')
        
        self.conn.commit()
    
    def store_interaction(self, metrics: InteractionMetrics):
        """Store interaction metrics in database"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.interaction_id,
            metrics.timestamp.isoformat(),
            metrics.user_type,
            metrics.strategy,
            metrics.query,
            metrics.response,
            metrics.confidence_score,
            metrics.trust_level,
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.total_tokens,
            metrics.response_time,
            metrics.cost_estimate,
            metrics.user_satisfaction,
            int(metrics.escalated),
            int(metrics.cache_hit)
        ))
        
        self.conn.commit()
    
    def get_recent_interactions(self, hours: int = 24) -> List[InteractionMetrics]:
        """Get recent interactions within specified time period"""
        cursor = self.conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute('''
            SELECT * FROM interactions 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        
        interactions = []
        for row in rows:
            interactions.append(InteractionMetrics(
                interaction_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                user_type=row[2],
                strategy=row[3],
                query=row[4],
                response=row[5],
                confidence_score=row[6],
                trust_level=row[7],
                input_tokens=row[8],
                output_tokens=row[9],
                total_tokens=row[10],
                response_time=row[11],
                cost_estimate=row[12],
                user_satisfaction=row[13],
                escalated=bool(row[14]),
                cache_hit=bool(row[15])
            ))
        
        return interactions

class RealTimeMonitor:
    """Real-time performance monitoring and alerting"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db = PerformanceDatabase(db_path)
        self.performance_targets = {
            'response_time': 2.0,
            'token_usage': 200,
            'cost_per_interaction': 0.05,
            'confidence_accuracy': 0.85,
            'escalation_rate': 0.15,
            'user_satisfaction': 0.80
        }
        
        self.alert_thresholds = {
            'response_time': 5.0,
            'token_usage': 500,
            'cost_per_interaction': 0.10,
            'confidence_accuracy': 0.70,
            'escalation_rate': 0.30,
            'user_satisfaction': 0.60
        }
    
    def record_interaction(self, metrics: InteractionMetrics):
        """Record interaction and check for alerts"""
        self.db.store_interaction(metrics)
        self._check_real_time_alerts(metrics)
    
    def _check_real_time_alerts(self, metrics: InteractionMetrics):
        """Check for immediate performance alerts"""
        alerts = []
        
        # Response time alert
        if metrics.response_time > self.alert_thresholds['response_time']:
            alerts.append({
                'type': 'response_time',
                'message': f"High response time: {metrics.response_time:.2f}s",
                'severity': 'warning'
            })
        
        # Token usage alert
        if metrics.total_tokens > self.alert_thresholds['token_usage']:
            alerts.append({
                'type': 'token_usage',
                'message': f"High token usage: {metrics.total_tokens} tokens",
                'severity': 'warning'
            })
        
        # Cost alert
        if metrics.cost_estimate > self.alert_thresholds['cost_per_interaction']:
            alerts.append({
                'type': 'cost',
                'message': f"High cost: ${metrics.cost_estimate:.4f}",
                'severity': 'warning'
            })
        
        # Store alerts in database
        for alert in alerts:
            self._store_alert(alert)
    
    def _store_alert(self, alert: Dict):
        """Store alert in database"""
        cursor = self.db.conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_alerts (timestamp, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            alert['type'],
            alert['message'],
            alert['severity']
        ))
        
        self.db.conn.commit()
    
    def get_performance_dashboard(self, hours: int = 24) -> Dict:
        """Generate comprehensive performance dashboard"""
        interactions = self.db.get_recent_interactions(hours)
        
        if not interactions:
            return {'error': 'No interactions found in specified time period'}
        
        return {
            'summary': self._calculate_summary_metrics(interactions),
            'strategy_performance': self._analyze_strategy_performance(interactions),
            'user_type_analysis': self._analyze_user_type_performance(interactions),
            'trust_level_distribution': self._analyze_trust_distribution(interactions),
            'performance_trends': self._calculate_trends(interactions),
            'alerts': self._get_active_alerts(),
            'recommendations': self._generate_recommendations(interactions)
        }
    
    def _calculate_summary_metrics(self, interactions: List[InteractionMetrics]) -> Dict:
        """Calculate summary performance metrics"""
        total_interactions = len(interactions)
        
        # Response time metrics
        response_times = [i.response_time for i in interactions]
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        # Token usage metrics
        token_usage = [i.total_tokens for i in interactions]
        avg_tokens = statistics.mean(token_usage)
        
        # Cost metrics
        costs = [i.cost_estimate for i in interactions]
        total_cost = sum(costs)
        avg_cost = statistics.mean(costs)
        
        # Confidence metrics
        confidence_scores = [i.confidence_score for i in interactions]
        avg_confidence = statistics.mean(confidence_scores)
        
        # Trust level distribution
        trust_levels = [i.trust_level for i in interactions]
        trust_distribution = {
            'auto_run': trust_levels.count('auto_run') / total_interactions,
            'confirm': trust_levels.count('confirm') / total_interactions,
            'escalate': trust_levels.count('escalate') / total_interactions
        }
        
        # User satisfaction (if available)
        satisfied_interactions = [i for i in interactions if i.user_satisfaction is not None]
        avg_satisfaction = statistics.mean([i.user_satisfaction for i in satisfied_interactions]) if satisfied_interactions else None
        
        return {
            'total_interactions': total_interactions,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'avg_tokens': avg_tokens,
            'total_cost': total_cost,
            'avg_cost': avg_cost,
            'avg_confidence': avg_confidence,
            'trust_distribution': trust_distribution,
            'avg_satisfaction': avg_satisfaction,
            'cache_hit_rate': sum(1 for i in interactions if i.cache_hit) / total_interactions,
            'escalation_rate': sum(1 for i in interactions if i.escalated) / total_interactions
        }
    
    def _analyze_strategy_performance(self, interactions: List[InteractionMetrics]) -> Dict:
        """Analyze performance by strategy"""
        strategy_groups = {}
        
        for interaction in interactions:
            strategy = interaction.strategy
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(interaction)
        
        strategy_analysis = {}
        for strategy, group in strategy_groups.items():
            strategy_analysis[strategy] = {
                'count': len(group),
                'avg_response_time': statistics.mean([i.response_time for i in group]),
                'avg_tokens': statistics.mean([i.total_tokens for i in group]),
                'avg_cost': statistics.mean([i.cost_estimate for i in group]),
                'avg_confidence': statistics.mean([i.confidence_score for i in group]),
                'escalation_rate': sum(1 for i in group if i.escalated) / len(group)
            }
        
        return strategy_analysis
    
    def _analyze_user_type_performance(self, interactions: List[InteractionMetrics]) -> Dict:
        """Analyze performance by user type"""
        user_groups = {}
        
        for interaction in interactions:
            user_type = interaction.user_type
            if user_type not in user_groups:
                user_groups[user_type] = []
            user_groups[user_type].append(interaction)
        
        user_analysis = {}
        for user_type, group in user_groups.items():
            satisfied_interactions = [i for i in group if i.user_satisfaction is not None]
            
            user_analysis[user_type] = {
                'count': len(group),
                'avg_response_time': statistics.mean([i.response_time for i in group]),
                'avg_tokens': statistics.mean([i.total_tokens for i in group]),
                'avg_satisfaction': statistics.mean([i.user_satisfaction for i in satisfied_interactions]) if satisfied_interactions else None,
                'preferred_strategies': self._get_preferred_strategies(group)
            }
        
        return user_analysis
    
    def _get_preferred_strategies(self, interactions: List[InteractionMetrics]) -> Dict:
        """Get preferred strategies for a group of interactions"""
        strategy_counts = {}
        for interaction in interactions:
            strategy = interaction.strategy
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        total = len(interactions)
        return {strategy: count / total for strategy, count in strategy_counts.items()}
    
    def _analyze_trust_distribution(self, interactions: List[InteractionMetrics]) -> Dict:
        """Analyze trust level distribution and accuracy"""
        trust_analysis = {}
        
        for trust_level in ['auto_run', 'confirm', 'escalate']:
            trust_interactions = [i for i in interactions if i.trust_level == trust_level]
            
            if trust_interactions:
                satisfied_interactions = [i for i in trust_interactions if i.user_satisfaction is not None]
                
                trust_analysis[trust_level] = {
                    'count': len(trust_interactions),
                    'percentage': len(trust_interactions) / len(interactions),
                    'avg_confidence': statistics.mean([i.confidence_score for i in trust_interactions]),
                    'avg_satisfaction': statistics.mean([i.user_satisfaction for i in satisfied_interactions]) if satisfied_interactions else None,
                    'avg_response_time': statistics.mean([i.response_time for i in trust_interactions])
                }
        
        return trust_analysis
    
    def _calculate_trends(self, interactions: List[InteractionMetrics]) -> Dict:
        """Calculate performance trends over time"""
        # Sort by timestamp
        interactions.sort(key=lambda x: x.timestamp)
        
        # Split into time buckets (hourly)
        hourly_buckets = {}
        for interaction in interactions:
            hour_key = interaction.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_buckets:
                hourly_buckets[hour_key] = []
            hourly_buckets[hour_key].append(interaction)
        
        trends = {}
        for hour, bucket in hourly_buckets.items():
            trends[hour.isoformat()] = {
                'interaction_count': len(bucket),
                'avg_response_time': statistics.mean([i.response_time for i in bucket]),
                'avg_tokens': statistics.mean([i.total_tokens for i in bucket]),
                'avg_confidence': statistics.mean([i.confidence_score for i in bucket])
            }
        
        return trends
    
    def _get_active_alerts(self) -> List[Dict]:
        """Get active performance alerts"""
        cursor = self.db.conn.cursor()
        
        cursor.execute('''
            SELECT * FROM performance_alerts 
            WHERE resolved = 0 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        rows = cursor.fetchall()
        
        alerts = []
        for row in rows:
            alerts.append({
                'alert_id': row[0],
                'timestamp': row[1],
                'type': row[2],
                'message': row[3],
                'severity': row[4]
            })
        
        return alerts
    
    def _generate_recommendations(self, interactions: List[InteractionMetrics]) -> List[str]:
        """Generate optimization recommendations based on performance data"""
        recommendations = []
        
        # Calculate key metrics
        avg_response_time = statistics.mean([i.response_time for i in interactions])
        avg_tokens = statistics.mean([i.total_tokens for i in interactions])
        avg_cost = statistics.mean([i.cost_estimate for i in interactions])
        escalation_rate = sum(1 for i in interactions if i.escalated) / len(interactions)
        
        # Response time recommendations
        if avg_response_time > self.performance_targets['response_time']:
            recommendations.append(f"Consider optimizing response time (current: {avg_response_time:.2f}s, target: {self.performance_targets['response_time']}s)")
        
        # Token usage recommendations
        if avg_tokens > self.performance_targets['token_usage']:
            recommendations.append(f"Consider implementing token compression (current: {avg_tokens:.0f}, target: {self.performance_targets['token_usage']})")
        
        # Cost optimization recommendations
        if avg_cost > self.performance_targets['cost_per_interaction']:
            recommendations.append(f"Consider cost optimization strategies (current: ${avg_cost:.4f}, target: ${self.performance_targets['cost_per_interaction']})")
        
        # Escalation rate recommendations
        if escalation_rate > self.performance_targets['escalation_rate']:
            recommendations.append(f"High escalation rate detected ({escalation_rate:.1%}). Consider improving confidence scoring or expanding knowledge base.")
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = RealTimeMonitor()
    
    # Create sample interaction
    sample_interaction = InteractionMetrics(
        interaction_id="test_001",
        timestamp=datetime.now(),
        user_type="customer",
        strategy="chain_of_thought",
        query="What are the pricing options?",
        response="Here are our pricing tiers...",
        confidence_score=0.85,
        trust_level="confirm",
        input_tokens=50,
        output_tokens=150,
        total_tokens=200,
        response_time=1.5,
        cost_estimate=0.03,
        user_satisfaction=0.9,
        escalated=False,
        cache_hit=False
    )
    
    # Record interaction
    monitor.record_interaction(sample_interaction)
    
    # Get dashboard
    dashboard = monitor.get_performance_dashboard(hours=24)
    print(json.dumps(dashboard, indent=2, default=str))
