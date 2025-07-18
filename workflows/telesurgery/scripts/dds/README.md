## Troubleshooting DDS Communication

### Common Issues and Solutions

- **Connection Problems**:
  - Ensure your RTI license is properly installed and valid
  - Verify the environment variable `RTI_LICENSE_FILE` points to the correct license file location

- **Topic Discovery Issues**:
  - Confirm that QoS (Quality of Service) settings match between publishers and subscribers
  - Ensure domain IDs are consistent across all communicating applications
  - Try increasing the discovery wait time in your application

- **Performance Optimization**:
  - For high-frequency topics (like camera feeds), adjust history depth to prevent buffer overflow
  - Consider using BEST_EFFORT reliability for streaming data where occasional packet loss is acceptable
  - Monitor resource usage and adjust queue sizes accordingly

For comprehensive troubleshooting guidance, refer to the [RTI Connext DDS Getting Started Guide](https://www.rti.com/gettingstarted).


### Discovery Issues

```bash
export PATH=$PATH:/opt/rti.com/rti_connext_dds-7.5.0/bin

rtiddsping -domainId 9 -publisher  -peer <peer_ip>
rtiddsping -domainId 9 -subscriber -peer <peer_ip>
```
