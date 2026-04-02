//+------------------------------------------------------------------+
//|                                                    signal_reader |
//+------------------------------------------------------------------+
struct SignalData
{
   datetime timestamp;
   int signal_type;
   double confidence;
   double sl;
   double tp;
   double atr;
};

bool ReadSignal(SignalData &signal)
{
   string path = "C:\\Users\\Ahmed\\Desktop\\gold_trading_system\\mt5_ea\\signal.txt";
   int file = FileOpen(path, FILE_READ|FILE_TXT);
   if(file == INVALID_HANDLE)
      return false;
   string content = FileReadString(file);
   FileClose(file);
   if(content == "" || StringLen(content) < 10)
      return false;
   
   signal.timestamp = TimeCurrent();
   signal.signal_type = 0;
   signal.confidence = 0;
   signal.sl = 0;
   signal.tp = 0;
   signal.atr = 10;
   
   // Parse timestamp
   int pos = StringFind(content, "\"timestamp\":");
   if(pos >= 0)
   {
      int start = pos + 13;
      int end = StringFind(content, ",", start);
      if(end > start)
      {
         string ts = StringSubstr(content, start, end - start);
         StringReplace(ts, "\"", "");
         signal.timestamp = StringToTime(ts);
      }
   }
   
   // Parse signal
   pos = StringFind(content, "\"signal\":");
   if(pos >= 0)
   {
      int start = pos + 10;
      int end = StringFind(content, ",", start);
      if(end > start)
      {
         string sig = StringSubstr(content, start, end - start);
         StringReplace(sig, "\"", "");
         StringReplace(sig, " ", "");
         if(sig == "buy") signal.signal_type = 1;
         else if(sig == "sell") signal.signal_type = 2;
      }
   }
   
   // Parse confidence
   pos = StringFind(content, "\"confidence\":");
   if(pos >= 0)
   {
      int start = pos + 13;
      int end = StringFind(content, ",", start);
      if(end > start)
         signal.confidence = StringToDouble(StringSubstr(content, start, end - start));
   }
   
   // Parse sl
   pos = StringFind(content, "\"sl\":");
   if(pos >= 0)
   {
      int start = pos + 5;
      int end = StringFind(content, ",", start);
      if(end > start)
         signal.sl = StringToDouble(StringSubstr(content, start, end - start));
   }
   
   // Parse tp
   pos = StringFind(content, "\"tp\":");
   if(pos >= 0)
   {
      int start = pos + 5;
      int end = StringFind(content, ",", start);
      if(end > start)
         signal.tp = StringToDouble(StringSubstr(content, start, end - start));
   }
   
   // Parse atr
   pos = StringFind(content, "\"atr\":");
   if(pos >= 0)
   {
      int start = pos + 6;
      int end = StringFind(content, "}", start);
      if(end <= 0) end = StringLen(content);
      if(end > start)
         signal.atr = StringToDouble(StringSubstr(content, start, end - start));
   }
   
   return (signal.signal_type > 0);
}
