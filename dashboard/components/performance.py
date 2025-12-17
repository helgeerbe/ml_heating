"""
ML Heating Dashboard - Performance Analytics Component
Advanced analytics for ML system performance and insights
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import pickle

# Add app directory to Python path
sys.path.append('/app')

def load_ml_analytics_data():
    """Load comprehensive ML analytics data"""
    try:
        # Try to load actual analytics if available
        analytics_file = '/data/models/ml_analytics.pkl'
        if os.path.exists(analytics_file):
            with open(analytics_file, 'rb') as f:
                return pickle.load(f)
    except Exception:
        pass
    
    # Generate demo analytics data for Phase 4 development
    return generate_demo_analytics()

def generate_demo_analytics():
    """Generate comprehensive demo analytics data"""
    np.random.seed(42)  # For reproducible demo data
    
    # Generate 30 days of learning data
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30), 
        end=datetime.now(), 
        freq='H'
    )
    
    analytics = {
        'learning_history': [],
        'feature_importance': {},
        'prediction_accuracy': [],
        'energy_efficiency': [],
        'weather_correlation': [],
        'system_insights': {}
    }
    
    # Learning history with realistic progression
    for i, date in enumerate(dates):
        # Simulate learning improvement over time
        base_confidence = min(0.95, 0.3 + (i / len(dates)) * 0.65)
        confidence = base_confidence + np.random.normal(0, 0.05)
        confidence = max(0.0, min(1.0, confidence))
        
        # MAE decreases as system learns
        base_mae = max(0.1, 0.8 - (i / len(dates)) * 0.6)
        mae = base_mae + np.random.normal(0, 0.1)
        mae = max(0.05, mae)
        
        analytics['learning_history'].append({
            'timestamp': date,
            'confidence': confidence,
            'mae': mae,
            'rmse': mae * 1.2,
            'cycle_count': i,
            'prediction_accuracy': min(98, 60 + (i / len(dates)) * 35)
        })
    
    # Feature importance (what factors influence predictions most)
    analytics['feature_importance'] = {
        'outdoor_temperature': 0.35,
        'indoor_temperature': 0.25,
        'time_of_day': 0.15,
        'solar_irradiance': 0.12,
        'wind_speed': 0.08,
        'humidity': 0.05
    }
    
    # Prediction accuracy over time
    for i in range(len(dates)):
        accuracy = 60 + (i / len(dates)) * 35 + np.random.normal(0, 3)
        analytics['prediction_accuracy'].append({
            'timestamp': dates[i],
            'accuracy_percent': min(98, max(50, accuracy)),
            'predictions_made': np.random.randint(20, 50),
            'correct_predictions': int((accuracy / 100) * np.random.randint(20, 50))
        })
    
    # Energy efficiency metrics
    baseline_consumption = 100  # kWh baseline
    for i, date in enumerate(dates[:24*7]):  # Weekly data
        # Efficiency improves as system learns
        efficiency_gain = min(25, (i / (24*7)) * 20)
        consumption = baseline_consumption * (1 - efficiency_gain/100)
        consumption += np.random.normal(0, 5)
        
        analytics['energy_efficiency'].append({
            'date': date.date(),
            'consumption_kwh': max(60, consumption),
            'baseline_kwh': baseline_consumption,
            'savings_percent': efficiency_gain,
            'cost_savings_eur': efficiency_gain * 0.25  # €0.25/kWh
        })
    
    # Weather correlation data
    for temp in range(-10, 30, 5):
        # Simulate how prediction accuracy varies with outdoor temperature
        if -5 <= temp <= 15:  # Optimal range
            accuracy = 85 + np.random.normal(0, 5)
        else:  # Extreme temperatures
            accuracy = 75 + np.random.normal(0, 8)
        
        analytics['weather_correlation'].append({
            'outdoor_temp': temp,
            'prediction_accuracy': min(95, max(60, accuracy)),
            'confidence_level': min(0.95, max(0.6, accuracy/100))
        })
    
    # System insights
    analytics['system_insights'] = {
        'total_learning_cycles': len(dates),
        'avg_confidence': np.mean([h['confidence'] for h in analytics['learning_history']]),
        'current_accuracy': analytics['prediction_accuracy'][-1]['accuracy_percent'],
        'energy_savings_total': sum([e['savings_percent'] for e in analytics['energy_efficiency']]) / len(analytics['energy_efficiency']),
        'optimal_temp_range': {'min': 18, 'max': 22},
        'learning_rate': 'Optimal',
        'recommendation': 'System performing well - continue current configuration'
    }
    
    return analytics

def render_learning_progress():
    """Render learning progress visualization"""
    st.subheader("📈 Learning Progress Over Time")
    
    analytics = load_ml_analytics_data()
    df = pd.DataFrame(analytics['learning_history'])
    
    if df.empty:
        st.warning("No learning data available yet.")
        return
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Confidence & Accuracy', 'Error Metrics', 
                       'Learning Cycles', 'Prediction Quality'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Confidence and accuracy
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['confidence'],
                  name='Confidence', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['prediction_accuracy'],
                  name='Accuracy %', line=dict(color='green')),
        row=1, col=1, secondary_y=True
    )
    
    # Error metrics
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['mae'],
                  name='MAE', line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['rmse'],
                  name='RMSE', line=dict(color='orange')),
        row=1, col=2
    )
    
    # Learning cycles
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['cycle_count'],
                  name='Cycles', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Prediction quality trend
    moving_avg = df['prediction_accuracy'].rolling(window=24).mean()
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=moving_avg,
                  name='24h Avg Accuracy', line=dict(color='darkgreen')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True,
                     title_text="ML System Learning Analytics")
    st.plotly_chart(fig, use_container_width=True)

def render_feature_importance():
    """Render feature importance analysis"""
    st.subheader("🎯 Feature Importance Analysis")
    st.caption("Which factors most influence the ML system's predictions")
    
    analytics = load_ml_analytics_data()
    features = analytics['feature_importance']
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    feature_names = list(features.keys())
    importances = list(features.values())
    
    # Color code by importance
    colors = ['#1f77b4' if imp > 0.2 else '#ff7f0e' if imp > 0.1 else '#2ca02c' 
             for imp in importances]
    
    fig.add_trace(go.Bar(
        y=feature_names,
        x=importances,
        orientation='h',
        marker_color=colors,
        text=[f'{imp:.1%}' for imp in importances],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Feature Importance in ML Predictions",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Key Insights:**")
        top_feature = max(features, key=features.get)
        st.write(f"• Most important: **{top_feature.replace('_', ' ').title()}** ({features[top_feature]:.1%})")
        st.write(f"• Top 3 factors account for **{sum(sorted(importances, reverse=True)[:3]):.1%}** of decisions")
        
    with col2:
        st.success("**Recommendations:**")
        if features['outdoor_temperature'] > 0.3:
            st.write("• Outdoor temp sensor is critical - ensure accuracy")
        if features['solar_irradiance'] > 0.1:
            st.write("• Solar data helps - consider weather integration")
        st.write("• System learning well from available data")

def render_prediction_accuracy():
    """Render prediction accuracy analysis"""
    st.subheader("🎯 Prediction Accuracy Analysis")
    
    analytics = load_ml_analytics_data()
    accuracy_data = pd.DataFrame(analytics['prediction_accuracy'])
    weather_data = pd.DataFrame(analytics['weather_correlation'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Accuracy Over Time**")
        
        # Accuracy trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=accuracy_data['timestamp'],
            y=accuracy_data['accuracy_percent'],
            mode='lines+markers',
            name='Prediction Accuracy',
            line=dict(color='green')
        ))
        
        # Add trend line
        x_numeric = np.arange(len(accuracy_data))
        z = np.polyfit(x_numeric, accuracy_data['accuracy_percent'], 1)
        trend_line = np.poly1d(z)(x_numeric)
        
        fig.add_trace(go.Scatter(
            x=accuracy_data['timestamp'],
            y=trend_line,
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            yaxis_title="Accuracy (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Weather Impact on Accuracy**")
        
        # Weather correlation
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weather_data['outdoor_temp'],
            y=weather_data['prediction_accuracy'],
            mode='markers+lines',
            name='Accuracy vs Temperature',
            marker=dict(color='blue', size=8)
        ))
        
        fig.update_layout(
            xaxis_title="Outdoor Temperature (°C)",
            yaxis_title="Accuracy (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Accuracy metrics
    current_accuracy = accuracy_data['accuracy_percent'].iloc[-1]
    avg_accuracy = accuracy_data['accuracy_percent'].mean()
    
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Current Accuracy", f"{current_accuracy:.1f}%", 
                 f"{current_accuracy - avg_accuracy:+.1f}%")
    with col4:
        st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
    with col5:
        optimal_range = weather_data[
            (weather_data['outdoor_temp'] >= -5) & 
            (weather_data['outdoor_temp'] <= 15)
        ]['prediction_accuracy'].mean()
        st.metric("Optimal Range Accuracy", f"{optimal_range:.1f}%")

def render_shadow_mode_benchmarks():
    """Render shadow mode ML vs Heat Curve benchmarking analysis"""
    st.subheader("🎯 Shadow Mode: ML vs Heat Curve Benchmarks")
    st.caption("Comparing ML predictions against heat curve performance in shadow mode")
    
    # Load shadow mode benchmark data (would come from InfluxDB in production)
    benchmark_data = generate_demo_shadow_benchmarks()
    
    if not benchmark_data:
        st.warning("Shadow mode benchmarking data not available. Enable SHADOW_MODE to collect benchmarks.")
        return
    
    df = pd.DataFrame(benchmark_data)
    
    # Main comparison chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Outlet Temperature Comparison', 'Efficiency Advantage Over Time',
                       'Target Achievement Accuracy', 'Energy Savings Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Outlet temperature comparison
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['ml_outlet_prediction'],
                  name='ML Prediction', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['heat_curve_outlet_actual'],
                  name='Heat Curve Actual', line=dict(color='red')),
        row=1, col=1
    )
    
    # Efficiency advantage over time
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['efficiency_advantage'],
                  name='Efficiency Advantage', 
                  line=dict(color='green'),
                  fill='tonexty' if df['efficiency_advantage'].mean() > 0 else None),
        row=1, col=2
    )
    
    # Target achievement accuracy
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['target_achievement_accuracy'],
                  name='Achievement Accuracy', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Energy savings distribution
    fig.add_trace(
        go.Histogram(x=df['energy_savings_pct'], name='Energy Savings %',
                    marker_color='lightgreen', nbinsx=20),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True,
                     title_text="Shadow Mode Benchmarking Analysis")
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Energy Savings (%)", row=2, col=2)
    fig.update_yaxes(title_text="Outlet Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Advantage (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    avg_ml_outlet = df['ml_outlet_prediction'].mean()
    avg_hc_outlet = df['heat_curve_outlet_actual'].mean()
    avg_efficiency = df['efficiency_advantage'].mean()
    avg_savings = df['energy_savings_pct'].mean()
    
    with col1:
        st.metric("Avg ML Outlet", f"{avg_ml_outlet:.1f}°C")
    with col2:
        st.metric("Avg Heat Curve", f"{avg_hc_outlet:.1f}°C")
    with col3:
        delta = avg_efficiency
        st.metric("Efficiency Advantage", f"{avg_efficiency:+.1f}°C", 
                 f"{'Lower outlet = more efficient' if delta < 0 else 'Higher outlet needed'}")
    with col4:
        st.metric("Energy Savings", f"{avg_savings:.1f}%",
                 f"{'Savings' if avg_savings > 0 else 'No savings'}")
    
    # Insights and recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Benchmarking Insights:**")
        if avg_efficiency < 0:
            st.write(f"• ML predicts **{abs(avg_efficiency):.1f}°C lower** outlet temps on average")
            st.write("• ML system shows potential for energy savings")
            st.write(f"• Estimated **{avg_savings:.1f}%** energy reduction possible")
        else:
            st.write(f"• Heat curve operates **{avg_efficiency:.1f}°C lower** than ML would predict")
            st.write("• Current heat curve already optimized")
            st.write("• Consider switching to active ML mode")
    
    with col2:
        if avg_efficiency < -2.0:
            st.success("**Recommendation: Switch to Active Mode**")
            st.write("• ML shows significant efficiency potential")
            st.write("• Shadow mode learning appears sufficient")
            st.write("• Consider enabling ML control")
        elif abs(avg_efficiency) < 1.0:
            st.warning("**Recommendation: Continue Shadow Mode**")
            st.write("• Performance similar between systems")
            st.write("• Allow more learning time")
            st.write("• Monitor trend development")
        else:
            st.info("**Recommendation: Review Configuration**")
            st.write("• Check heat curve settings")
            st.write("• Verify sensor calibration")
            st.write("• Monitor learning progress")

def generate_demo_shadow_benchmarks():
    """Generate demo shadow mode benchmark data"""
    np.random.seed(42)
    
    # Generate 7 days of shadow mode benchmark data
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq='H'
    )
    
    benchmark_data = []
    
    for i, ts in enumerate(timestamps):
        # Simulate outdoor temperature cycle
        hour_of_day = ts.hour
        outdoor_temp = 5 + 10 * np.sin((hour_of_day - 6) * np.pi / 12) + np.random.normal(0, 2)
        
        # Heat curve typically sets higher outlet temps
        heat_curve_outlet = 35 + (10 - outdoor_temp) * 0.8 + np.random.normal(0, 1)
        heat_curve_outlet = max(25, min(55, heat_curve_outlet))
        
        # ML learns to be more efficient over time
        learning_progress = min(1.0, i / (len(timestamps) * 0.7))
        ml_efficiency_gain = learning_progress * 3.5  # Up to 3.5°C lower
        ml_outlet = heat_curve_outlet - ml_efficiency_gain + np.random.normal(0, 0.5)
        ml_outlet = max(20, min(50, ml_outlet))
        
        # Calculate derived metrics
        efficiency_advantage = heat_curve_outlet - ml_outlet  # Positive = ML more efficient
        energy_savings_pct = max(0, efficiency_advantage * 2.5)  # Rough conversion
        
        # Target achievement accuracy (both systems good at maintaining temperature)
        target_accuracy = 85 + learning_progress * 10 + np.random.normal(0, 3)
        target_accuracy = max(70, min(98, target_accuracy))
        
        benchmark_data.append({
            'timestamp': ts,
            'ml_outlet_prediction': ml_outlet,
            'heat_curve_outlet_actual': heat_curve_outlet,
            'efficiency_advantage': efficiency_advantage,
            'energy_savings_pct': energy_savings_pct,
            'target_achievement_accuracy': target_accuracy,
            'outdoor_temp': outdoor_temp,
            'learning_progress': learning_progress * 100
        })
    
    return benchmark_data

def render_energy_efficiency():
    """Render energy efficiency analysis"""
    st.subheader("⚡ Energy Efficiency Analysis")
    st.caption("Measuring ML system's impact on energy consumption")
    
    analytics = load_ml_analytics_data()
    efficiency_data = pd.DataFrame(analytics['energy_efficiency'])
    
    if efficiency_data.empty:
        st.warning("Energy efficiency data not available yet.")
        return
    
    # Main efficiency chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Energy Consumption', 'Cost Savings'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Consumption comparison
    fig.add_trace(
        go.Scatter(x=efficiency_data['date'], y=efficiency_data['baseline_kwh'],
                  name='Baseline', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=efficiency_data['date'], y=efficiency_data['consumption_kwh'],
                  name='ML Optimized', line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=efficiency_data['date'], y=efficiency_data['savings_percent'],
                  name='Savings %', line=dict(color='blue')),
        row=1, col=1, secondary_y=True
    )
    
    # Cost savings
    fig.add_trace(
        go.Bar(x=efficiency_data['date'], y=efficiency_data['cost_savings_eur'],
               name='Daily Savings (€)', marker_color='green'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Savings (%)", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Cost Savings (€)", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_savings = efficiency_data['savings_percent'].mean()
    total_cost_savings = efficiency_data['cost_savings_eur'].sum()
    avg_consumption = efficiency_data['consumption_kwh'].mean()
    baseline_avg = efficiency_data['baseline_kwh'].iloc[0]
    
    with col1:
        st.metric("Average Savings", f"{total_savings:.1f}%")
    with col2:
        st.metric("Total Cost Savings", f"€{total_cost_savings:.2f}")
    with col3:
        st.metric("Avg Consumption", f"{avg_consumption:.1f} kWh")
    with col4:
        reduction = ((baseline_avg - avg_consumption) / baseline_avg) * 100
        st.metric("Energy Reduction", f"{reduction:.1f}%")


def render_system_insights():
    """Render AI-generated system insights and recommendations"""
    st.subheader("🤖 System Insights & Recommendations")
    
    analytics = load_ml_analytics_data()
    insights = analytics['system_insights']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Performance Summary**")
        st.write(f"• **Learning Cycles**: {insights['total_learning_cycles']:,}")
        st.write(f"• **Average Confidence**: {insights['avg_confidence']:.1%}")
        st.write(f"• **Current Accuracy**: {insights['current_accuracy']:.1f}%")
        st.write(f"• **Energy Savings**: {insights['energy_savings_total']:.1f}%")
        
        # Performance rating
        if insights['avg_confidence'] > 0.9:
            st.success("🌟 **Performance: Excellent**")
        elif insights['avg_confidence'] > 0.8:
            st.info("👍 **Performance: Good**")
        else:
            st.warning("⚠️ **Performance: Needs Improvement**")
    
    with col2:
        st.success("**AI Recommendations**")
        st.write(f"• **Learning Rate**: {insights['learning_rate']}")
        
        temp_range = insights['optimal_temp_range']
        st.write(f"• **Optimal Indoor Range**: {temp_range['min']}-{temp_range['max']}°C")
        
        st.write(f"• **System Status**: {insights['recommendation']}")
        
        # Dynamic recommendations based on performance
        if insights['avg_confidence'] < 0.8:
            st.write("• Consider increasing learning rate")
            st.write("• Check sensor calibration")
        if insights['energy_savings_total'] < 10:
            st.write("• Review temperature targets")
            st.write("• Verify heating control integration")


def render_seasonal_analysis():
    """Render seasonal performance analysis"""
    st.subheader("🌡️ Seasonal Performance Analysis")
    
    # Generate seasonal demo data
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasonal_data = {
        'Winter': {'accuracy': 82, 'efficiency': 18, 'challenges': 'Extreme cold'},
        'Spring': {'accuracy': 88, 'efficiency': 22, 'challenges': 'Variable weather'},
        'Summer': {'accuracy': 75, 'efficiency': 12, 'challenges': 'Minimal heating needed'},
        'Autumn': {'accuracy': 85, 'efficiency': 20, 'challenges': 'Transition period'}
    }
    
    # Create seasonal comparison chart
    fig = go.Figure()
    
    accuracies = [seasonal_data[season]['accuracy'] for season in seasons]
    efficiencies = [seasonal_data[season]['efficiency'] for season in seasons]
    
    fig.add_trace(go.Bar(
        name='Accuracy (%)',
        x=seasons,
        y=accuracies,
        yaxis='y1',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        name='Efficiency (%)',
        x=seasons,
        y=efficiencies,
        yaxis='y2',
        marker_color='green'
    ))
    
    fig.update_layout(
        title="Seasonal Performance Comparison",
        xaxis_title="Season",
        yaxis=dict(title="Accuracy (%)", side="left"),
        yaxis2=dict(title="Efficiency (%)", side="right", overlaying="y"),
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Seasonal Challenges:**")
        for season, data in seasonal_data.items():
            st.write(f"• **{season}**: {data['challenges']}")
    
    with col2:
        st.write("**Best Performance:**")
        best_accuracy_season = max(seasonal_data, key=lambda x: seasonal_data[x]['accuracy'])
        best_efficiency_season = max(seasonal_data, key=lambda x: seasonal_data[x]['efficiency'])
        
        st.success(f"🎯 **Accuracy**: {best_accuracy_season}")
        st.success(f"⚡ **Efficiency**: {best_efficiency_season}")


def render_performance():
    """Main performance analytics page"""
    st.header("📈 Performance Analytics")
    st.caption("Advanced ML system performance analysis and insights")
    
    # Auto-refresh option
    if st.button("🔄 Refresh Analytics"):
        st.experimental_rerun()
    
    # Render all analytics sections
    render_learning_progress()
    
    st.divider()
    
    render_feature_importance()
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_prediction_accuracy()
    
    with col2:
        render_energy_efficiency()
    
    st.divider()
    
    # Add shadow mode benchmarking section
    render_shadow_mode_benchmarks()
    
    st.divider()
    
    render_system_insights()
    
    st.divider()
    
    render_seasonal_analysis()
    
    # Export options
    st.divider()
    st.subheader("📊 Export Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Report (PDF)"):
            st.info("PDF export functionality coming soon")
    
    with col2:
        if st.button("📊 Export Data (CSV)"):
            st.info("CSV export functionality coming soon")
    
    with col3:
        if st.button("📈 Share Dashboard"):
            st.info("Dashboard sharing functionality coming soon")
