from extract import DataExtractor

def main():
    extractor = DataExtractor(r"C:\Users\Majo\source\repos\TorchCompresser\cmd\gate_info\data")
    df = extractor.extract()
    print(df)
    extractor.save(df)

if __name__ == "__main__":
    main()
    