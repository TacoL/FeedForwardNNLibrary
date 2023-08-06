namespace FeedForwardNNLibrary
{
    public struct TrainingSample
    {
        public double[] inputs, targets;
        public TrainingSample(double[] inp, double[] tar)
        {
            inputs = inp;
            targets = tar;
        }
    }
}
