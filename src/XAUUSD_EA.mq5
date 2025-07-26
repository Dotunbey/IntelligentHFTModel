// MQL5 Expert Advisor for simulating HFT trades with ONNX model
#property strict
#include <Files\FileCsv.mqh>

void OnInit() {
    Print("HFT Model Initialized (Simulation with Sample Data)");
}

void OnTick() {
    // Read trades from CSV (simulates ONNX model output)
    CFileCsv file;
    if(file.Open("trades.csv", FILE_READ|FILE_CSV)) {
        string action = file.ReadString();
        double price = StringToDouble(file.ReadString());
        double cx7 = StringToDouble(file.ReadString());
        double sl = StringToDouble(file.ReadString());
        double order_flow_imbalance = StringToDouble(file.ReadString());
        double sentiment = StringToDouble(file.ReadString());
        if(cx7 > 0.5 && order_flow_imbalance > 0.5 && sentiment > 0.2 && action == "buy")
            Print("Simulated Buy: Price=", price, ", cx7=", cx7, ", Imbalance=", order_flow_imbalance, ", Sentiment=", sentiment, ", SL=", sl);
        else if(cx7 < -0.5 && order_flow_imbalance < -0.5 && sentiment < -0.2 && action == "sell")
            Print("Simulated Sell: Price=", price, ", cx7=", cx7, ", Imbalance=", order_flow_imbalance, ", Sentiment=", sentiment, ", SL=", sl);
        file.Close();
    }
}
