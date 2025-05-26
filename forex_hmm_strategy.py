import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import MT5 data provider
try:
    from mt5_data_provider import MT5DataProvider
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️  MT5 not available, will use fallback data")

class ForexHMMStrategy:
    def __init__(self, symbol='EURUSD', n_components=3, risk_reward_ratio=1.5, use_mt5=True):
        """
        Initialize the HMM-based Forex trading strategy
        
        Parameters:
        symbol: Currency pair symbol (EURUSD, GBPUSD, etc. for MT5 or EUR=X for Yahoo)
        n_components: Number of hidden states in HMM
        risk_reward_ratio: Risk to reward ratio (1:1.5 means 1.5)
        use_mt5: Whether to use MT5 data (True) or Yahoo Finance (False)
        """
        self.symbol = symbol
        self.n_components = n_components
        self.risk_reward_ratio = risk_reward_ratio
        self.use_mt5 = use_mt5 and MT5_AVAILABLE
        self.model = None
        self.scaler = StandardScaler()
        self.data = None
        self.features = None
        self.states = None
        self.signals = None
        self.mt5_provider = None
        
        # Initialize MT5 if requested
        if self.use_mt5:
            self.mt5_provider = MT5DataProvider()
            print("🔄 Initializing MT5 connection...")
        
    def fetch_data(self, period='1y', interval='5m'):
        """
        Fetch forex data from MT5 or Yahoo Finance
        """
        print(f"Fetching {self.symbol} data...")
        
        # Try MT5 first if available
        if self.use_mt5 and self.mt5_provider:
            try:
                if not self.mt5_provider.connected:
                    if not self.mt5_provider.connect():
                        print("❌ Failed to connect to MT5, falling back to Yahoo Finance")
                        self.use_mt5 = False
                    
                if self.use_mt5:
                    # Convert period to number of bars
                    bars_map = {
                        '1d': 288,    # 1 day = 288 5-min bars
                        '1w': 2016,   # 1 week = 2016 5-min bars  
                        '1mo': 8640,  # 1 month ≈ 8640 5-min bars
                        '3mo': 25920, # 3 months ≈ 25920 5-min bars
                        '1y': 103680  # 1 year ≈ 103680 5-min bars
                    }
                    bars = bars_map.get(period, 8640)  # Default to 1 month
                    
                    # Fetch data from MT5
                    self.data = self.mt5_provider.fetch_data(
                        symbol=self.symbol, 
                        timeframe=mt5.TIMEFRAME_M5,
                        bars=bars
                    )
                    
                    if self.data is not None and len(self.data) > 100:
                        print(f"✅ Successfully fetched {len(self.data)} bars from MT5")
                        return self.data
                    else:
                        print("❌ MT5 data fetch failed, falling back to Yahoo Finance")
                        self.use_mt5 = False
                        
            except Exception as e:
                print(f"❌ MT5 error: {e}, falling back to Yahoo Finance")
                self.use_mt5 = False
        
        # Fallback to Yahoo Finance
        print("📊 Using Yahoo Finance data...")
        
        # Convert MT5 symbol to Yahoo format
        yahoo_symbol = self.symbol
        if self.symbol == 'EURUSD':
            yahoo_symbol = 'EUR=X'
        elif self.symbol == 'GBPUSD':
            yahoo_symbol = 'GBP=X'
        elif self.symbol == 'USDJPY':
            yahoo_symbol = 'JPY=X'
        
        # Try multiple symbol formats for better compatibility
        symbols_to_try = [yahoo_symbol, self.symbol]
        if yahoo_symbol == 'EUR=X':
            symbols_to_try = ['EUR=X', 'EURUSD=X', 'EURUSD']
        elif yahoo_symbol == 'GBP=X':
            symbols_to_try = ['GBP=X', 'GBPUSD=X', 'GBPUSD']
        elif yahoo_symbol == 'JPY=X':
            symbols_to_try = ['JPY=X', 'USDJPY=X', 'USDJPY']
        
        self.data = None
        for symbol in symbols_to_try:
            try:
                print(f"Trying Yahoo symbol: {symbol}")
                data = yf.download(symbol, period=period, interval=interval, progress=False)
                if not data.empty and len(data) > 100:  # Ensure we have sufficient data
                    self.data = data
                    print(f"✅ Successfully fetched data using symbol: {symbol}")
                    break
            except Exception as e:
                print(f"Failed with {symbol}: {e}")
                continue
        
        # If forex symbols don't work, try a major stock as fallback for demonstration
        if self.data is None or self.data.empty:
            print("Forex data not available, using SPY (S&P 500 ETF) as demonstration...")
            try:
                self.data = yf.download('SPY', period=period, interval=interval, progress=False)
                print("✅ Using SPY data for demonstration")
            except Exception as e:
                print(f"Failed to fetch SPY data: {e}")
        
        if self.data is None or self.data.empty:
            raise ValueError(f"No data could be fetched for any symbol. Please check your internet connection.")
            
        # Clean data
        self.data = self.data.dropna()
        print(f"Data fetched: {len(self.data)} records for {self.symbol}")
        return self.data
    
    def create_features(self):
        """
        Create technical indicators and features for HMM
        """
        df = self.data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_change'] = df['Close'] - df['Open']
        df['high_low_ratio'] = df['High'] / df['Low']
        df['volume_price'] = df['Volume'] * df['Close']
        
        # Technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['bb_upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['bb_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['Close']
        
        # Moving averages
        df['sma_5'] = ta.trend.SMAIndicator(df['Close'], window=5).sma_indicator()
        df['sma_10'] = ta.trend.SMAIndicator(df['Close'], window=10).sma_indicator()
        df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['ema_5'] = ta.trend.EMAIndicator(df['Close'], window=5).ema_indicator()
        df['ema_10'] = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
        
        # Price position relative to moving averages
        df['price_sma5_ratio'] = df['Close'] / df['sma_5']
        df['price_sma10_ratio'] = df['Close'] / df['sma_10']
        df['price_sma20_ratio'] = df['Close'] / df['sma_20']
        
        # Volatility features
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Momentum features
        df['momentum'] = df['Close'] / df['Close'].shift(10)
        df['roc'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
        
        # Volume features (if available)
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            try:
                # Try different volume indicators
                df['volume_sma'] = df['Volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['Volume'] / df['volume_sma']
            except:
                df['volume_ratio'] = 1.0
        else:
            df['volume_ratio'] = 1.0
        
        # Time-based features for scalping
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        
        # Market session indicators (useful for forex)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['tokyo_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        
        # Select features for HMM - use robust feature set
        # Start with essential features that are less likely to cause convergence issues
        essential_features = [
            'returns', 'log_returns', 'price_change', 
            'rsi', 'price_sma5_ratio', 'price_sma10_ratio',
            'atr', 'volatility', 'momentum'
        ]
        
        # Add additional features if data quality is good
        additional_features = [
            'high_low_ratio', 'macd', 'bb_width', 'roc', 'volume_ratio',
            'london_session', 'ny_session', 'overlap_session'
        ]
        
        # Check which features have sufficient data quality
        feature_columns = essential_features.copy()
        
        for feature in additional_features:
            if feature in df.columns:
                feature_data = df[feature].dropna()
                # Only include if feature has good data coverage and variance
                if len(feature_data) > len(df) * 0.8 and feature_data.std() > 1e-6:
                    feature_columns.append(feature)
                else:
                    print(f"⚠️ Excluding feature {feature} due to poor data quality")
        
        print(f"Using {len(feature_columns)} features for HMM: {feature_columns}")
        
        self.features = df[feature_columns].dropna()
        self.data = df.loc[self.features.index]
        
        print(f"Features created: {self.features.shape}")
        return self.features
    
    def train_hmm(self):
        """
        Train the Hidden Markov Model with improved convergence handling
        """
        if self.features is None:
            raise ValueError("Features not created. Call create_features() first.")
        
        # Clean and validate features
        features_clean = self.features.copy()
        
        # Remove infinite and NaN values
        features_clean = features_clean.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.dropna()
        
        if len(features_clean) < 100:
            raise ValueError(f"Insufficient clean data for HMM training: {len(features_clean)} samples")
        
        print(f"Training HMM with {len(features_clean)} clean samples...")
        
        # Scale features with robust scaling
        features_scaled = self.scaler.fit_transform(features_clean)
        
        # Check for constant features (zero variance)
        feature_std = np.std(features_scaled, axis=0)
        valid_features = feature_std > 1e-6
        
        if not np.all(valid_features):
            print(f"⚠️ Removing {np.sum(~valid_features)} constant features")
            features_scaled = features_scaled[:, valid_features]
            # Update feature names for tracking
            valid_feature_names = features_clean.columns[valid_features].tolist()
            print(f"Using features: {valid_feature_names}")
        
        # Try multiple HMM configurations for better convergence
        hmm_configs = [
            {"covariance_type": "diag", "n_iter": 50},
            {"covariance_type": "spherical", "n_iter": 50}, 
            {"covariance_type": "full", "n_iter": 50},
            {"covariance_type": "diag", "n_iter": 100},
            {"covariance_type": "tied", "n_iter": 50}
        ]
        
        best_model = None
        best_score = -np.inf
        
        for config in hmm_configs:
            try:
                print(f"Trying HMM config: {config}")
                
                model = hmm.GaussianHMM(
                    n_components=self.n_components,
                    covariance_type=config["covariance_type"],
                    n_iter=config["n_iter"],
                    random_state=42,
                    tol=1e-3,  # Relaxed tolerance
                    verbose=False
                )
                
                # Fit with error handling
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(features_scaled)
                
                # Check if model converged
                if hasattr(model, 'monitor_') and hasattr(model.monitor_, 'converged'):
                    if not model.monitor_.converged:
                        print(f"⚠️ Model did not converge with {config}")
                        continue
                
                # Calculate model score (likelihood)
                try:
                    score = model.score(features_scaled)
                    print(f"✅ Model score: {score:.2f}")
                    
                    if score > best_score and not np.isnan(score) and not np.isinf(score):
                        best_score = score
                        best_model = model
                        print(f"🎯 New best model found!")
                        
                except Exception as e:
                    print(f"❌ Error scoring model: {e}")
                    continue
                    
            except Exception as e:
                print(f"❌ Error training HMM with {config}: {e}")
                continue
        
        if best_model is None:
            print("❌ All HMM configurations failed, using fallback simple model")
            # Fallback to simplest possible model
            try:
                self.model = hmm.GaussianHMM(
                    n_components=2,  # Reduce to 2 states
                    covariance_type="spherical",
                    n_iter=20,
                    random_state=42,
                    tol=1e-2  # Very relaxed tolerance
                )
                
                # Use only the most stable features for fallback
                stable_features = features_scaled[:, :3]  # Use first 3 features only
                self.model.fit(stable_features)
                self.states = self.model.predict(stable_features)
                
                # Store the feature names for fallback model
                self.trained_feature_names = list(features_clean.columns[:3])
                print(f"✅ Fallback HMM model trained with 2 states using features: {self.trained_feature_names}")
                
            except Exception as e:
                print(f"❌ Even fallback model failed: {e}")
                # Ultimate fallback - create dummy model
                self.create_dummy_model(features_scaled)
                return self.model
        else:
            self.model = best_model
            # Predict hidden states
            self.states = self.model.predict(features_scaled)
            print(f"✅ Best HMM model trained with {self.n_components} states (score: {best_score:.2f})")
        
        # Update features to match cleaned data
        self.features = features_clean
        self.data = self.data.loc[features_clean.index]
        
        # Store the feature names that were used for training
        self.trained_feature_names = list(features_clean.columns)
        print(f"✅ Stored trained feature names: {self.trained_feature_names}")
        
        return self.model
    
    def create_dummy_model(self, features_scaled):
        """Create a dummy model when HMM training completely fails"""
        print("🔄 Creating dummy model as ultimate fallback...")
        
        # Create a simple rule-based "model"
        class DummyHMM:
            def __init__(self, n_components=3):
                self.n_components = n_components
                
            def predict(self, X):
                # Simple rule: use price momentum to determine state
                if len(X) > 0:
                    # Use first feature (usually returns) to determine state
                    returns = X[:, 0] if X.shape[1] > 0 else np.zeros(len(X))
                    states = np.zeros(len(returns), dtype=int)
                    
                    # Assign states based on return quantiles
                    q33 = np.percentile(returns, 33)
                    q67 = np.percentile(returns, 67)
                    
                    states[returns <= q33] = 0  # Bearish
                    states[(returns > q33) & (returns <= q67)] = 1  # Neutral
                    states[returns > q67] = 2  # Bullish
                    
                    return states
                return np.zeros(1, dtype=int)
            
            def predict_proba(self, X):
                # Return uniform probabilities
                states = self.predict(X)
                probs = np.ones((len(states), self.n_components)) / self.n_components
                return probs
        
        self.model = DummyHMM(self.n_components)
        self.states = self.model.predict(features_scaled)
        
        # Store feature names for dummy model (use first 3 features)
        if hasattr(self, 'features') and self.features is not None:
            self.trained_feature_names = list(self.features.columns[:3])
        else:
            # Fallback feature names
            self.trained_feature_names = ['returns', 'log_returns', 'price_change']
        
        print(f"✅ Dummy model created successfully using features: {self.trained_feature_names}")
    
    def retrain_for_symbol(self, symbol, period='3mo'):
        """Retrain the model for a specific symbol when feature mismatch occurs"""
        print(f"🔄 Retraining HMM model for symbol: {symbol}")
        
        # Update symbol
        old_symbol = self.symbol
        self.symbol = symbol
        
        try:
            # Fetch new data for the symbol
            self.fetch_data(period=period, interval='5m')
            
            # Create features
            self.create_features()
            
            # Train HMM
            self.train_hmm()
            
            print(f"✅ Model successfully retrained for {symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to retrain model for {symbol}: {e}")
            # Restore old symbol
            self.symbol = old_symbol
            return False
    
    def prepare_features(self, df):
        """Prepare features from OHLC data for prediction - MUST match training features"""
        try:
            # Create a temporary copy for feature generation
            temp_df = df.copy()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in temp_df.columns]
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return np.array([])
            
            # Add Volume column if missing (for compatibility)
            if 'Volume' not in temp_df.columns:
                temp_df['Volume'] = 1.0  # Dummy volume
                print("⚠️ Volume column missing, using dummy values")
            
            print(f"🔧 Input DataFrame shape: {temp_df.shape}")
            print(f"🔧 Input DataFrame columns: {list(temp_df.columns)}")
            print(f"🔧 Input DataFrame index type: {type(temp_df.index)}")
            
            # COMPLETE FEATURE SET - matching create_features() exactly
            
            # Price-based features
            temp_df['returns'] = temp_df['Close'].pct_change()
            temp_df['log_returns'] = np.log(temp_df['Close'] / temp_df['Close'].shift(1))
            temp_df['price_change'] = temp_df['Close'] - temp_df['Open']
            temp_df['high_low_ratio'] = temp_df['High'] / temp_df['Low']
            
            # Volume-price feature (with fallback)
            if 'Volume' in temp_df.columns:
                temp_df['volume_price'] = temp_df['Volume'] * temp_df['Close']
            else:
                temp_df['volume_price'] = temp_df['Close']  # Fallback to just price
            
            # Technical indicators using TA library
            try:
                import ta
                temp_df['rsi'] = ta.momentum.RSIIndicator(temp_df['Close'], window=14).rsi()
                temp_df['macd'] = ta.trend.MACD(temp_df['Close']).macd()
                temp_df['macd_signal'] = ta.trend.MACD(temp_df['Close']).macd_signal()
                temp_df['bb_upper'] = ta.volatility.BollingerBands(temp_df['Close']).bollinger_hband()
                temp_df['bb_lower'] = ta.volatility.BollingerBands(temp_df['Close']).bollinger_lband()
                temp_df['bb_width'] = (temp_df['bb_upper'] - temp_df['bb_lower']) / temp_df['Close']
                temp_df['atr'] = ta.volatility.AverageTrueRange(temp_df['High'], temp_df['Low'], temp_df['Close']).average_true_range()
                temp_df['roc'] = ta.momentum.ROCIndicator(temp_df['Close'], window=10).roc()
            except ImportError:
                # Fallback calculations without TA library
                # Simple RSI
                delta = temp_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                temp_df['rsi'] = 100 - (100 / (1 + rs))
                
                # Simple MACD
                ema12 = temp_df['Close'].ewm(span=12).mean()
                ema26 = temp_df['Close'].ewm(span=26).mean()
                temp_df['macd'] = ema12 - ema26
                temp_df['macd_signal'] = temp_df['macd'].ewm(span=9).mean()
                
                # Simple Bollinger Bands
                sma20 = temp_df['Close'].rolling(20).mean()
                std20 = temp_df['Close'].rolling(20).std()
                temp_df['bb_upper'] = sma20 + (std20 * 2)
                temp_df['bb_lower'] = sma20 - (std20 * 2)
                temp_df['bb_width'] = (temp_df['bb_upper'] - temp_df['bb_lower']) / temp_df['Close']
                
                # Simple ATR
                temp_df['high_low'] = temp_df['High'] - temp_df['Low']
                temp_df['atr'] = temp_df['high_low'].rolling(14).mean()
                
                # Simple ROC
                temp_df['roc'] = temp_df['Close'].pct_change(10) * 100
            
            # Moving averages
            temp_df['sma_5'] = temp_df['Close'].rolling(5).mean()
            temp_df['sma_10'] = temp_df['Close'].rolling(10).mean()
            temp_df['sma_20'] = temp_df['Close'].rolling(20).mean()
            temp_df['ema_5'] = temp_df['Close'].ewm(span=5).mean()
            temp_df['ema_10'] = temp_df['Close'].ewm(span=10).mean()
            
            # Price position relative to moving averages
            temp_df['price_sma5_ratio'] = temp_df['Close'] / temp_df['sma_5']
            temp_df['price_sma10_ratio'] = temp_df['Close'] / temp_df['sma_10']
            temp_df['price_sma20_ratio'] = temp_df['Close'] / temp_df['sma_20']
            
            # Volatility features
            temp_df['volatility'] = temp_df['returns'].rolling(window=20).std()
            
            # Momentum features
            temp_df['momentum'] = temp_df['Close'] / temp_df['Close'].shift(10)
            
            # Volume features (if available)
            if 'Volume' in temp_df.columns and temp_df['Volume'].sum() > 0:
                temp_df['volume_sma'] = temp_df['Volume'].rolling(window=20).mean()
                temp_df['volume_ratio'] = temp_df['Volume'] / temp_df['volume_sma']
            else:
                temp_df['volume_ratio'] = 1.0
            
            # Time-based features for scalping (with proper datetime index handling)
            try:
                # Check if index is datetime-like
                if hasattr(temp_df.index, 'hour'):
                    temp_df['hour'] = temp_df.index.hour
                    temp_df['minute'] = temp_df.index.minute
                    temp_df['day_of_week'] = temp_df.index.dayofweek
                else:
                    # Create dummy time features if no datetime index
                    print("⚠️ No datetime index found, using dummy time features")
                    temp_df['hour'] = 12  # Default to noon (active trading time)
                    temp_df['minute'] = 0
                    temp_df['day_of_week'] = 1  # Default to Tuesday (mid-week)
                
                # Market session indicators (useful for forex)
                temp_df['london_session'] = ((temp_df['hour'] >= 8) & (temp_df['hour'] < 16)).astype(int)
                temp_df['ny_session'] = ((temp_df['hour'] >= 13) & (temp_df['hour'] < 21)).astype(int)
                temp_df['tokyo_session'] = ((temp_df['hour'] >= 0) & (temp_df['hour'] < 8)).astype(int)
                temp_df['overlap_session'] = ((temp_df['hour'] >= 13) & (temp_df['hour'] < 16)).astype(int)
                
            except Exception as time_error:
                print(f"⚠️ Error creating time features: {time_error}")
                # Fallback: create dummy time features
                temp_df['hour'] = 12  # Default to noon
                temp_df['minute'] = 0
                temp_df['day_of_week'] = 1
                temp_df['london_session'] = 1  # Default to active session
                temp_df['ny_session'] = 1
                temp_df['tokyo_session'] = 0
                temp_df['overlap_session'] = 1
            
            # If we have trained feature names, use them directly
            if hasattr(self, 'trained_feature_names') and self.trained_feature_names:
                print(f"🎯 Using stored trained feature names: {self.trained_feature_names}")
                feature_columns = self.trained_feature_names.copy()
            else:
                # Use the SAME feature selection logic as create_features()
                print("🔧 No trained feature names found, using default feature selection")
                essential_features = [
                    'returns', 'log_returns', 'price_change', 
                    'rsi', 'price_sma5_ratio', 'price_sma10_ratio',
                    'atr', 'volatility', 'momentum'
                ]
                
                additional_features = [
                    'high_low_ratio', 'macd', 'bb_width', 'roc', 'volume_ratio',
                    'london_session', 'ny_session', 'overlap_session'
                ]
                
                # Check which features have sufficient data quality (same logic as training)
                feature_columns = essential_features.copy()
                
                for feature in additional_features:
                    if feature in temp_df.columns:
                        feature_data = temp_df[feature].dropna()
                        # Only include if feature has good data coverage and variance
                        if len(feature_data) > len(temp_df) * 0.8 and feature_data.std() > 1e-6:
                            feature_columns.append(feature)
            
            # Filter to available features and remove NaN
            available_features = [f for f in feature_columns if f in temp_df.columns]
            
            print(f"🔧 Feature columns requested: {feature_columns}")
            print(f"🔧 Available features: {available_features}")
            
            if not available_features:
                print("❌ No features available after filtering")
                return np.array([])
            
            feature_df = temp_df[available_features].dropna()
            
            print(f"🔧 Feature DataFrame shape after dropna: {feature_df.shape}")
            
            if len(feature_df) == 0:
                print("⚠️ No valid features could be prepared after removing NaN")
                print("🔧 Trying with forward fill...")
                # Try forward fill to handle NaN values
                feature_df = temp_df[available_features].fillna(method='ffill').dropna()
                
                if len(feature_df) == 0:
                    print("❌ Still no valid features after forward fill")
                    return np.array([])
            
            print(f"🔧 Final prepared features: {len(available_features)} features, {len(feature_df)} samples")
            print(f"🔧 Feature names: {available_features}")
            
            # Scale features using the same scaler
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    # Check if we have the trained feature names
                    if hasattr(self, 'trained_feature_names'):
                        print(f"🔧 Model expects features: {self.trained_feature_names}")
                        print(f"🔧 Available features: {available_features}")
                        
                        # Ensure we have exactly the same features in the same order
                        missing_features = [f for f in self.trained_feature_names if f not in available_features]
                        extra_features = [f for f in available_features if f not in self.trained_feature_names]
                        
                        if missing_features:
                            print(f"❌ Missing trained features: {missing_features}")
                            return np.array([])
                        
                        if extra_features:
                            print(f"⚠️ Extra features (will be ignored): {extra_features}")
                        
                        # Select only the trained features in the correct order
                        feature_df_ordered = feature_df[self.trained_feature_names]
                        print(f"✅ Using exact feature match: {list(feature_df_ordered.columns)}")
                        
                        features_scaled = self.scaler.transform(feature_df_ordered)
                        return features_scaled
                    else:
                        # Fallback: try with available features
                        print("⚠️ No trained feature names stored, trying with available features")
                        features_scaled = self.scaler.transform(feature_df)
                        return features_scaled
                        
                except ValueError as e:
                    if "feature names" in str(e).lower():
                        print(f"❌ Feature mismatch error: {e}")
                        print(f"   Model expects features that were used during training")
                        print(f"   Available features: {available_features}")
                        if hasattr(self, 'trained_feature_names'):
                            print(f"   Trained features: {self.trained_feature_names}")
                        print(f"   This suggests the model needs retraining for current data")
                        return np.array([])
                    else:
                        print(f"⚠️ Error scaling features: {e}")
                        # Return unscaled features as fallback
                        return feature_df.values
            else:
                # No scaler available, return raw features
                return feature_df.values
                
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return np.array([])
    
    def generate_signals(self):
        """
        Generate trading signals based on HMM states and probabilities
        """
        if self.model is None or self.states is None:
            raise ValueError("Model not trained. Call train_hmm() first.")
        
        # Get state probabilities
        features_scaled = self.scaler.transform(self.features)
        state_probs = self.model.predict_proba(features_scaled)
        
        # Calculate state statistics
        state_returns = {}
        state_volatility = {}
        
        for state in range(self.n_components):
            state_mask = self.states == state
            if state_mask.sum() > 0:
                state_returns[state] = self.data.loc[self.features.index[state_mask], 'returns'].mean()
                state_volatility[state] = self.data.loc[self.features.index[state_mask], 'returns'].std()
        
        # Identify bullish and bearish states
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        bearish_state = sorted_states[0][0]
        bullish_state = sorted_states[-1][0]
        
        # Generate signals
        signals = pd.DataFrame(index=self.features.index)
        signals['state'] = self.states
        signals['bullish_prob'] = state_probs[:, bullish_state]
        signals['bearish_prob'] = state_probs[:, bearish_state]
        signals['current_state_prob'] = [state_probs[i, self.states[i]] for i in range(len(self.states))]
        
        # Signal generation logic for scalping
        signals['signal'] = 0
        
        # Buy signals: High probability of bullish state + additional conditions
        buy_condition = (
            (signals['bullish_prob'] > 0.6) &
            (signals['current_state_prob'] > 0.7) &
            (self.data.loc[signals.index, 'rsi'] < 70) &
            (self.data.loc[signals.index, 'rsi'] > 30) &
            (self.data.loc[signals.index, 'overlap_session'] == 1)  # Trade during overlap
        )
        
        # Sell signals: High probability of bearish state + additional conditions
        sell_condition = (
            (signals['bearish_prob'] > 0.6) &
            (signals['current_state_prob'] > 0.7) &
            (self.data.loc[signals.index, 'rsi'] > 30) &
            (self.data.loc[signals.index, 'rsi'] < 70) &
            (self.data.loc[signals.index, 'overlap_session'] == 1)  # Trade during overlap
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        # Add stop loss and take profit levels
        atr_values = self.data.loc[signals.index, 'atr']
        signals['stop_loss_pips'] = atr_values * 1.5  # Dynamic stop loss based on ATR
        signals['take_profit_pips'] = signals['stop_loss_pips'] * self.risk_reward_ratio
        
        self.signals = signals
        print(f"Signals generated: {(signals['signal'] != 0).sum()} total signals")
        return signals
    
    def generate_signals_optimized(self, prob_threshold=0.6, state_prob_threshold=0.7, atr_multiplier=1.5):
        """
        Generate trading signals with optimized parameters
        """
        if self.model is None or self.states is None:
            raise ValueError("Model not trained. Call train_hmm() first.")
        
        # Get state probabilities
        features_scaled = self.scaler.transform(self.features)
        state_probs = self.model.predict_proba(features_scaled)
        
        # Calculate state statistics
        state_returns = {}
        state_volatility = {}
        
        for state in range(self.n_components):
            state_mask = self.states == state
            if state_mask.sum() > 0:
                state_returns[state] = self.data.loc[self.features.index[state_mask], 'returns'].mean()
                state_volatility[state] = self.data.loc[self.features.index[state_mask], 'returns'].std()
        
        # Identify bullish and bearish states
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        bearish_state = sorted_states[0][0]
        bullish_state = sorted_states[-1][0]
        
        # Generate signals
        signals = pd.DataFrame(index=self.features.index)
        signals['state'] = self.states
        signals['bullish_prob'] = state_probs[:, bullish_state]
        signals['bearish_prob'] = state_probs[:, bearish_state]
        signals['current_state_prob'] = [state_probs[i, self.states[i]] for i in range(len(self.states))]
        
        # Signal generation logic with optimized thresholds
        signals['signal'] = 0
        
        # Buy signals with optimized conditions
        buy_condition = (
            (signals['bullish_prob'] > prob_threshold) &
            (signals['current_state_prob'] > state_prob_threshold) &
            (self.data.loc[signals.index, 'rsi'] < 70) &
            (self.data.loc[signals.index, 'rsi'] > 30) &
            (self.data.loc[signals.index, 'overlap_session'] == 1)
        )
        
        # Sell signals with optimized conditions
        sell_condition = (
            (signals['bearish_prob'] > prob_threshold) &
            (signals['current_state_prob'] > state_prob_threshold) &
            (self.data.loc[signals.index, 'rsi'] > 30) &
            (self.data.loc[signals.index, 'rsi'] < 70) &
            (self.data.loc[signals.index, 'overlap_session'] == 1)
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        # Add stop loss and take profit levels with optimized ATR multiplier
        atr_values = self.data.loc[signals.index, 'atr']
        signals['stop_loss_pips'] = atr_values * atr_multiplier
        signals['take_profit_pips'] = signals['stop_loss_pips'] * self.risk_reward_ratio
        
        self.signals = signals
        print(f"Optimized signals generated: {(signals['signal'] != 0).sum()} total signals")
        return signals
    
    def backtest(self, initial_capital=10000, position_size=0.1):
        """
        Backtest the strategy
        """
        if self.signals is None:
            raise ValueError("Signals not generated. Call generate_signals() first.")
        
        results = []
        capital = initial_capital
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trades = []
        
        for i, (timestamp, row) in enumerate(self.signals.iterrows()):
            current_price = self.data.loc[timestamp, 'Close']
            
            # Check for exit conditions if in position
            if position != 0:
                # Check stop loss
                if (position == 1 and current_price <= stop_loss) or \
                   (position == -1 and current_price >= stop_loss):
                    # Stop loss hit
                    pnl = (current_price - entry_price) * position * position_size
                    capital += pnl
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'exit_reason': 'stop_loss'
                    })
                    position = 0
                
                # Check take profit
                elif (position == 1 and current_price >= take_profit) or \
                     (position == -1 and current_price <= take_profit):
                    # Take profit hit
                    pnl = (current_price - entry_price) * position * position_size
                    capital += pnl
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'exit_reason': 'take_profit'
                    })
                    position = 0
            
            # Check for new entry signals
            if position == 0 and row['signal'] != 0:
                position = row['signal']
                entry_price = current_price
                entry_time = timestamp
                
                # Set stop loss and take profit
                if position == 1:  # Long position
                    stop_loss = entry_price - row['stop_loss_pips'] * 0.0001  # Convert pips to price
                    take_profit = entry_price + row['take_profit_pips'] * 0.0001
                else:  # Short position
                    stop_loss = entry_price + row['stop_loss_pips'] * 0.0001
                    take_profit = entry_price - row['take_profit_pips'] * 0.0001
            
            results.append({
                'timestamp': timestamp,
                'price': current_price,
                'signal': row['signal'],
                'position': position,
                'capital': capital,
                'drawdown': (capital - initial_capital) / initial_capital
            })
        
        # Close any remaining position
        if position != 0:
            current_price = self.data.loc[self.signals.index[-1], 'Close']
            pnl = (current_price - entry_price) * position * position_size
            capital += pnl
            trades.append({
                'entry_time': entry_time,
                'exit_time': self.signals.index[-1],
                'entry_price': entry_price,
                'exit_price': current_price,
                'position': position,
                'pnl': pnl,
                'exit_reason': 'end_of_data'
            })
        
        self.backtest_results = pd.DataFrame(results)
        self.trades = pd.DataFrame(trades)
        
        return self.analyze_performance()
    
    def analyze_performance(self):
        """
        Analyze backtest performance
        """
        if self.trades.empty:
            return {"error": "No trades executed"}
        
        # Calculate metrics
        total_trades = len(self.trades)
        winning_trades = len(self.trades[self.trades['pnl'] > 0])
        losing_trades = len(self.trades[self.trades['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = self.trades['pnl'].sum()
        avg_win = self.trades[self.trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = self.trades[self.trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Risk metrics
        returns = self.backtest_results['capital'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 12) if returns.std() != 0 else 0  # Annualized for 5-min data
        
        max_drawdown = self.backtest_results['drawdown'].min()
        
        performance = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': self.backtest_results['capital'].iloc[-1]
        }
        
        return performance
    
    def plot_results(self):
        """
        Plot backtest results
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Price and signals
        axes[0].plot(self.data.index, self.data['Close'], label='Price', alpha=0.7)
        buy_signals = self.signals[self.signals['signal'] == 1]
        sell_signals = self.signals[self.signals['signal'] == -1]
        
        axes[0].scatter(buy_signals.index, self.data.loc[buy_signals.index, 'Close'], 
                       color='green', marker='^', s=50, label='Buy Signal')
        axes[0].scatter(sell_signals.index, self.data.loc[sell_signals.index, 'Close'], 
                       color='red', marker='v', s=50, label='Sell Signal')
        axes[0].set_title('Price and Trading Signals')
        axes[0].legend()
        
        # HMM States
        axes[1].scatter(self.signals.index, self.signals['state'], 
                       c=self.signals['state'], cmap='viridis', alpha=0.6)
        axes[1].set_title('HMM Hidden States')
        axes[1].set_ylabel('State')
        
        # Capital curve
        axes[2].plot(self.backtest_results['timestamp'], self.backtest_results['capital'])
        axes[2].set_title('Capital Curve')
        axes[2].set_ylabel('Capital')
        
        # Drawdown
        axes[3].fill_between(self.backtest_results['timestamp'], 
                           self.backtest_results['drawdown'], 0, alpha=0.3, color='red')
        axes[3].set_title('Drawdown')
        axes[3].set_ylabel('Drawdown %')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def cleanup(self):
        """Clean up resources"""
        if self.mt5_provider and self.mt5_provider.connected:
            self.mt5_provider.disconnect()

def main():
    """
    Main function to run the strategy
    """
    strategy = None
    try:
        # Initialize strategy with MT5 (using available symbol)
        strategy = ForexHMMStrategy(symbol='BTCUSDm', n_components=3, risk_reward_ratio=1.5, use_mt5=True)
        
        # Fetch data
        strategy.fetch_data(period='3mo', interval='5m')  # 3 months of 5-minute data for scalping
        
        # Create features
        strategy.create_features()
        
        # Train HMM
        strategy.train_hmm()
        
        # Generate signals
        strategy.generate_signals()
        
        # Backtest
        performance = strategy.backtest(initial_capital=10000, position_size=0.1)
        
        # Print results
        print("\n" + "="*50)
        print("FOREX HMM SCALPING STRATEGY RESULTS")
        print("="*50)
        print(f"Data Source: {'MT5' if strategy.use_mt5 else 'Yahoo Finance'}")
        print(f"Symbol: {strategy.symbol}")
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Win Rate: {performance['win_rate']:.2%}")
        print(f"Total PnL: ${performance['total_pnl']:.2f}")
        print(f"Profit Factor: {performance['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"Final Capital: ${performance['final_capital']:.2f}")
        print("="*50)
        
        # Plot results
        strategy.plot_results()
        
        return strategy, performance
        
    except Exception as e:
        print(f"❌ Error in main: {e}")
        return None, None
    finally:
        # Clean up MT5 connection
        if strategy:
            strategy.cleanup()

if __name__ == "__main__":
    strategy, performance = main() 