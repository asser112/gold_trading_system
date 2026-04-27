//+------------------------------------------------------------------+
//|                                               XAUUSD_Trading_EA.mq5 |
//|                                                          Trading Bot |
//+------------------------------------------------------------------+
#property copyright "Trading Bot"
#property link      ""
#property version   "2.10"

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input group "=== Risk Management ==="
input double   RiskPercent      = 1.0;       // Risk per trade (%)
input double   MaxDailyLoss     = 5.0;        // Max daily loss (%)
input int      MaxPositions     = 1;          // Max simultaneous positions
input double   MaxLotSize       = 1.0;        // Maximum lot size

input group "=== Indicators ==="
input int      EMA_Fast         = 8;          // Fast EMA period
input int      EMA_Slow         = 21;          // Slow EMA period
input int      EMA_Trend        = 50;          // Trend EMA period
input int      RSI_Period       = 14;          // RSI period
input int      ATR_Period       = 14;          // ATR period
input double   SL_ATR_Mult      = 2.0;        // Stop Loss ATR multiplier
input double   TP_ATR_Mult      = 3.0;        // Take Profit ATR multiplier
input int      MinRRRatio       = 150;         // Minimum Risk:Reward (x100)

input group "=== Trading ==="
input int      MagicNumber       = 123456;     // EA Magic Number
input ulong    Deviation         = 5;          // Order deviation in points

input group "=== Signal API ==="
input string   SignalUrl        = "https://gold.yepwoo.com/api/signal";  // Backend signal URL
input string   ApiKey           = "admin-test-key-00000000";              // Your API key from dashboard
input string   SignalFileName   = "";   // (Legacy) Local file fallback — leave blank when using API

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CTrade         trade;
datetime       lastTradeTime    = 0;
double         dailyLoss        = 0;
double         peakBalance      = 0;
datetime       lastResetTime    = 0;
string         g_symbol         = "";
datetime       g_lastSignalRead = 0;

int            h_emaFast;
int            h_emaSlow;
int            h_emaTrend;
int            h_rsi;
int            h_atr;
int            h_stoch;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(Deviation);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   g_symbol = Symbol();
   peakBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   lastResetTime = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
   
   h_emaFast  = iMA(g_symbol, PERIOD_CURRENT, EMA_Fast,  0, MODE_EMA, PRICE_CLOSE);
   h_emaSlow  = iMA(g_symbol, PERIOD_CURRENT, EMA_Slow,  0, MODE_EMA, PRICE_CLOSE);
   h_emaTrend = iMA(g_symbol, PERIOD_CURRENT, EMA_Trend, 0, MODE_EMA, PRICE_CLOSE);
   h_rsi      = iRSI(g_symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
   h_atr      = iATR(g_symbol, PERIOD_CURRENT, ATR_Period);
   h_stoch    = iStochastic(g_symbol, PERIOD_CURRENT, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   
   if(h_emaFast == INVALID_HANDLE || h_emaSlow == INVALID_HANDLE ||
      h_emaTrend == INVALID_HANDLE || h_rsi == INVALID_HANDLE ||
      h_atr == INVALID_HANDLE || h_stoch == INVALID_HANDLE)
   {
      Print("Failed to create indicator handles");
      return(INIT_FAILED);
   }
   
   Print("EA initialized. Symbol: ", g_symbol);
   if(SignalUrl != "" && ApiKey != "")
      Print("Signal source: API (", SignalUrl, ")");
   else if(SignalFileName != "")
      Print("Signal source: local file (", SignalFileName, ")");
   else
      Print("Signal source: built-in indicators");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(h_emaFast);
   IndicatorRelease(h_emaSlow);
   IndicatorRelease(h_emaTrend);
   IndicatorRelease(h_rsi);
   IndicatorRelease(h_atr);
   IndicatorRelease(h_stoch);
}

//+------------------------------------------------------------------+
//| Check new bar                                                     |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   datetime currentTime = iTime(g_symbol, PERIOD_CURRENT, 0);
   if(currentTime != lastTradeTime)
   {
      lastTradeTime = currentTime;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Count open positions                                              |
//+------------------------------------------------------------------+
int PositionsCount()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == g_symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Calculate daily P&L                                              |
//+------------------------------------------------------------------+
void UpdateDailyStats()
{
   datetime currentDate = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
   
   if(currentDate != lastResetTime)
   {
      dailyLoss = 0;
      peakBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      lastResetTime = currentDate;
   }
   else
   {
      double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      dailyLoss = peakBalance - currentBalance;
      if(currentBalance > peakBalance)
         peakBalance = currentBalance;
   }
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                       |
//+------------------------------------------------------------------+
bool CanTrade()
{
   UpdateDailyStats();
   
   double dailyLossPercent = (dailyLoss / peakBalance) * 100;
   double marginLevel      = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   int    openPositions    = PositionsCount();
   double balance          = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity           = AccountInfoDouble(ACCOUNT_EQUITY);

   Print("[CHECK] Balance=", DoubleToString(balance,2),
         " Equity=", DoubleToString(equity,2),
         " DailyLoss%=", DoubleToString(dailyLossPercent,2),
         " Positions=", openPositions,
         " MarginLevel%=", DoubleToString(marginLevel,1));

   if(dailyLossPercent >= MaxDailyLoss)
   {
      Print("[BLOCK] Daily loss limit reached: ", DoubleToString(dailyLossPercent,2), "% >= ", MaxDailyLoss, "%");
      return false;
   }
   if(openPositions >= MaxPositions)
   {
      Print("[BLOCK] Max positions reached: ", openPositions, " >= ", MaxPositions);
      return false;
   }
   if(marginLevel > 0 && marginLevel < 150)
   {
      Print("[BLOCK] Low margin level: ", DoubleToString(marginLevel,1), "% < 150%");
      return false;
   }
   
   Print("[CHECK] CanTrade = true");
   return true;
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                   |
//+------------------------------------------------------------------+
double CalculateLotSize(double stopLossPoints)
{
   if(stopLossPoints <= 0)
      return 0.1;
       
   double riskAmount = AccountInfoDouble(ACCOUNT_BALANCE) * (RiskPercent / 100.0);
   double tickValue = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickSize <= 0) tickSize = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
   double pointValue = tickValue / tickSize;
   double lotSize = riskAmount / (stopLossPoints * pointValue);
   
   lotSize = lotSize / 100.0;
   
   double minLot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_STEP);
   if(lotStep <= 0) lotStep = 0.01;
   
   lotSize = MathMax(minLot, MathMin(lotSize, maxLot));
   lotSize = MathMin(lotSize, MaxLotSize);
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   
   return NormalizeDouble(lotSize, 2);
}

//+------------------------------------------------------------------+
//| Get indicator buffer value                                        |
//+------------------------------------------------------------------+
double GetIndicatorValue(int handle, int shift)
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(handle, 0, shift, 1, buf) == 1)
      return buf[0];
   return EMPTY_VALUE;
}

//+------------------------------------------------------------------+
//| Get indicator values                                              |
//+------------------------------------------------------------------+
double GetEMA(int handle, int shift)
{
   return GetIndicatorValue(handle, shift);
}

double GetRSI(int shift)
{
   return GetIndicatorValue(h_rsi, shift);
}

double GetATR(int shift)
{
   return GetIndicatorValue(h_atr, shift);
}

//+------------------------------------------------------------------+
//| Check for buy signal                                              |
//+------------------------------------------------------------------+
bool CheckBuySignal()
{
   double emaFast1 = GetEMA(h_emaFast, 1);
   double emaSlow1 = GetEMA(h_emaSlow, 1);
   double emaFast2 = GetEMA(h_emaFast, 2);
   double emaSlow2 = GetEMA(h_emaSlow, 2);
   double emaTrend = GetEMA(h_emaTrend, 1);
   double rsi = GetRSI(1);
   double close = iClose(g_symbol, PERIOD_CURRENT, 1);
   
   if(emaFast1 == EMPTY_VALUE || emaSlow1 == EMPTY_VALUE ||
      emaFast2 == EMPTY_VALUE || emaSlow2 == EMPTY_VALUE ||
      emaTrend == EMPTY_VALUE || rsi == EMPTY_VALUE)
      return false;
   
   bool emaCross = (emaFast1 > emaSlow1) && (emaFast2 <= emaSlow2);
   bool uptrend = close > emaTrend;
   bool rsiOk = rsi > 30 && rsi < 70;
   
   double stochK = GetIndicatorValue(h_stoch, 1);
   bool stochOk = (stochK != EMPTY_VALUE && stochK < 80);
   
   return emaCross && uptrend && rsiOk && stochOk;
}

//+------------------------------------------------------------------+
//| Check for sell signal                                             |
//+------------------------------------------------------------------+
bool CheckSellSignal()
{
   double emaFast1 = GetEMA(h_emaFast, 1);
   double emaSlow1 = GetEMA(h_emaSlow, 1);
   double emaFast2 = GetEMA(h_emaFast, 2);
   double emaSlow2 = GetEMA(h_emaSlow, 2);
   double emaTrend = GetEMA(h_emaTrend, 1);
   double rsi = GetRSI(1);
   double close = iClose(g_symbol, PERIOD_CURRENT, 1);
   
   if(emaFast1 == EMPTY_VALUE || emaSlow1 == EMPTY_VALUE ||
      emaFast2 == EMPTY_VALUE || emaSlow2 == EMPTY_VALUE ||
      emaTrend == EMPTY_VALUE || rsi == EMPTY_VALUE)
      return false;
   
   bool emaCross = (emaFast1 < emaSlow1) && (emaFast2 >= emaSlow2);
   bool downtrend = close < emaTrend;
   bool rsiOk = rsi < 70 && rsi > 30;
   
   double stochK = GetIndicatorValue(h_stoch, 1);
   bool stochOk = (stochK != EMPTY_VALUE && stochK > 20);
   
   return emaCross && downtrend && rsiOk && stochOk;
}

//+------------------------------------------------------------------+
//| Fetch signal from backend API (primary)                          |
//+------------------------------------------------------------------+
string FetchSignalFromAPI()
{
   if(SignalUrl == "" || ApiKey == "")
      return "";

   datetime now = TimeCurrent();
   if(now - g_lastSignalRead < 10)
      return "";
   g_lastSignalRead = now;

   string url = SignalUrl + "?api_key=" + ApiKey;
   string headers = "Content-Type: application/json\r\n";
   char   post[], result[];
   string resultHeaders;

   int httpCode = WebRequest("GET", url, headers, 5000, post, result, resultHeaders);
   if(httpCode != 200)
   {
      Print("[SIGNAL] API returned HTTP ", httpCode, ". Check ApiKey and subscription.");
      return "";
   }

   string json = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   Print("[SIGNAL] API response (", StringLen(json), " chars): ", json);
   return json;
}

//+------------------------------------------------------------------+
//| Read signal from local file (legacy fallback)                    |
//+------------------------------------------------------------------+
string ReadSignalFile()
{
   if(SignalFileName == "")
      return "";

   datetime now = TimeCurrent();
   if(now - g_lastSignalRead < 10)
      return "";
   g_lastSignalRead = now;

   int handle = FileOpen(SignalFileName, FILE_READ | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("[SIGNAL] Cannot open file: ", SignalFileName, " err=", GetLastError());
      return "";
   }

   string content = "";
   while(!FileIsEnding(handle))
      content += FileReadString(handle);
   FileClose(handle);

   if(StringLen(content) == 0)
      return "";
   return content;
}

//+------------------------------------------------------------------+
//| Get signal JSON from API (preferred) or file (fallback)          |
//+------------------------------------------------------------------+
string GetSignalJson()
{
   string json = FetchSignalFromAPI();
   if(json != "")
      return json;
   return ReadSignalFile();
}

//+------------------------------------------------------------------+
//| Parse signal from JSON content                                    |
//+------------------------------------------------------------------+
int ParseSignal(string json)
{
   if(json == "")
      return 0;
   
   // Find "signal" key
   int sigPos = StringFind(json, "\"signal\"", 0);
   if(sigPos == -1)
   {
      Print("[SIGNAL] No 'signal' key found in JSON");
      return 0;
   }
   
   // Find colon after "signal"
   int colonPos = StringFind(json, ":", sigPos);
   if(colonPos == -1)
   {
      Print("[SIGNAL] No colon after 'signal'");
      return 0;
   }
   
   // Find opening quote of value after colon
   int openQuote = -1;
   for(int i = colonPos + 1; i < StringLen(json); i++)
   {
      if(StringGetCharacter(json, i) == '\"')
      {
         openQuote = i;
         break;
      }
   }
   
   if(openQuote == -1)
   {
      Print("[SIGNAL] No opening quote for signal value");
      return 0;
   }
   
   // Find closing quote
   int closeQuote = -1;
   for(int i = openQuote + 1; i < StringLen(json); i++)
   {
      if(StringGetCharacter(json, i) == '\"')
      {
         closeQuote = i;
         break;
      }
   }
   
   if(closeQuote == -1)
   {
      Print("[SIGNAL] No closing quote for signal value");
      return 0;
   }
   
   string value = StringSubstr(json, openQuote + 1, closeQuote - openQuote - 1);
   value = StringLower(value);
   
   Print("[SIGNAL] Parsed value='", value, "'");
   
   if(value == "buy")
      return 1;
   if(value == "sell")
      return -1;
   if(value == "close" || value == "exit" || value == "hold")
      return 0;
   
   Print("[SIGNAL] Unknown value: ", value);
   return 0;
}

//+------------------------------------------------------------------+
//| String to lower case                                              |
//+------------------------------------------------------------------+
string StringLower(string s)
{
   string result = "";
   for(int i = 0; i < StringLen(s); i++)
   {
      ushort ch = StringGetCharacter(s, i);
      if(ch >= 'A' && ch <= 'Z')
         ch = ch - 'A' + 'a';
      result += ShortToString(ch);
   }
   return result;
}

//+------------------------------------------------------------------+
//| Close all positions                                               |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == g_symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber)
      {
         trade.PositionClose(PositionGetTicket(i));
      }
   }
}

//+------------------------------------------------------------------+
//| Main trading logic                                                |
//+------------------------------------------------------------------+
void ExecuteTradingLogic()
{
   if(!CanTrade())
      return;
   
   double atr = GetATR(1);
   if(atr == EMPTY_VALUE || atr <= 0)
   {
      Print("[BLOCK] ATR invalid: ", atr);
      return;
   }

   double ask   = SymbolInfoDouble(g_symbol, SYMBOL_ASK);
   double bid   = SymbolInfoDouble(g_symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
   int    digits = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
   double minLot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MIN);

   double slDistance = atr * SL_ATR_Mult / point;
   double tpDistance = atr * TP_ATR_Mult / point;
   if(tpDistance < slDistance * (MinRRRatio / 100.0))
      tpDistance = slDistance * (MinRRRatio / 100.0);

   double lot = CalculateLotSize(slDistance);

   Print("[INFO] ATR=", DoubleToString(atr,5),
         " Ask=", DoubleToString(ask,digits),
         " Bid=", DoubleToString(bid,digits),
         " SLdist=", DoubleToString(slDistance,1),
         " TPdist=", DoubleToString(tpDistance,1),
         " Lot=", DoubleToString(lot,2),
         " MinLot=", DoubleToString(minLot,2));

   int signal = 0;

   if(SignalUrl != "" || SignalFileName != "")
   {
      string json = GetSignalJson();
      if(json != "")
      {
         signal = ParseSignal(json);
         Print("[SIGNAL] Final result: ", signal, " (1=buy, -1=sell, 0=hold)");
      }
      else
         Print("[SIGNAL] No signal from API/file — falling back to indicators");
   }
   else
      Print("[SIGNAL] No URL or file configured — using built-in indicators");

   if(signal == 0)
   {
      if(CheckBuySignal())       { signal = 1;  Print("[INDICATOR] Buy signal from EMA"); }
      else if(CheckSellSignal()) { signal = -1; Print("[INDICATOR] Sell signal from EMA"); }
      else                         Print("[INDICATOR] No indicator signal — holding");
   }

   if(signal == 1)
   {
      double sl = NormalizeDouble(ask - atr * SL_ATR_Mult, digits);
      double tp = NormalizeDouble(ask + tpDistance * point, digits);
      if(lot >= minLot)
      {
         trade.Buy(lot, g_symbol, ask, sl, tp, "EA_Buy");
         Print("[TRADE] BUY Lot=", lot, " Price=", ask, " SL=", sl, " TP=", tp);
      }
      else
         Print("[BLOCK] Lot too small to buy: ", lot, " < minLot=", minLot);
   }
   else if(signal == -1)
   {
      double sl = NormalizeDouble(bid + atr * SL_ATR_Mult, digits);
      double tp = NormalizeDouble(bid - tpDistance * point, digits);
      if(lot >= minLot)
      {
         trade.Sell(lot, g_symbol, bid, sl, tp, "EA_Sell");
         Print("[TRADE] SELL Lot=", lot, " Price=", bid, " SL=", sl, " TP=", tp);
      }
      else
         Print("[BLOCK] Lot too small to sell: ", lot, " < minLot=", minLot);
   }
   else
      Print("[INFO] Signal=0 (hold) — no trade opened");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!IsNewBar())
      return;
      
   ExecuteTradingLogic();
}

//+------------------------------------------------------------------+
//| Expert timer function                                             |
//+------------------------------------------------------------------+
void OnTimer()
{
}
//+------------------------------------------------------------------+
